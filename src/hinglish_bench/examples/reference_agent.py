"""Reference agent backed by an in-memory MockDB.

The six mock tools (lookup_order, initiate_refund, check_emi_status,
reschedule_emi, track_delivery, escalate_to_human) are pure functions that
mutate a MockDB instance passed in at construction time.

ReferenceAgent.respond() calls its Role (which may be MockProvider or a real
LLM) to decide which tools to call, then executes them. There is no special-
casing for MockProvider — the agent always follows the same code path.
"""

from __future__ import annotations

import json

from hinglish_bench.agent import AgentTurn, ToolDef
from hinglish_bench.providers.base import ChatMessage, Role
from hinglish_bench.schemas import ToolCall, Turn

# Safety cap on tool-call rounds within a single agent turn, so a model that loops
# on tool calls cannot spin forever.
MAX_TOOL_ROUNDS = 6

# ------------------------------------------------------------------ #
# Mock database                                                        #
# ------------------------------------------------------------------ #

AGENT_SYSTEM_PROMPT = """\
You are a helpful customer service agent for an Indian e-commerce and lending platform.
You help callers with refunds, EMI scheduling, and delivery queries.
Use the available tools to look up information and take actions.
Reply in the same language as the caller. Be concise and clear.
"""


class MockDB:
    """In-memory database. Each run gets its own fresh instance via MockDB()."""

    def __init__(self) -> None:
        self.orders: dict[str, dict] = {
            # existing scenarios
            "ORD-88412": {
                "id": "ORD-88412",
                "product": "Floral kurta",
                "amount": 1499,
                "status": "delivered",
                "upi": "meera.iyer@okaxis",
            },
            "ORD-77301": {
                "id": "ORD-77301",
                "product": "Men's jeans",
                "amount": 899,
                "status": "in_transit",
                "upi": None,
            },
            # refund scenarios
            "ORD-11001": {
                "id": "ORD-11001",
                "product": "Cotton shirt (M)",
                "amount": 899,
                "status": "delivered",
                "upi": "priya.v@oksbi",
            },
            "ORD-11002": {
                "id": "ORD-11002",
                "product": "Running shoes",
                "amount": 2499,
                "status": "lost",
                "upi": "karthik.n@okhdfc",
            },
            "ORD-11003": {
                "id": "ORD-11003",
                "product": "Phone case",
                "amount": 349,
                "status": "delivered",
                "upi": "sunita.s@paytm",
            },
            "ORD-11004": {
                "id": "ORD-11004",
                "product": "Silk scarf",
                "amount": 599,
                "status": "delivered",
                "upi": "amit.k@phonepe",
            },
            "ORD-11005": {
                "id": "ORD-11005",
                "product": "Winter jacket",
                "amount": 1199,
                "status": "cancelled",
                "upi": "deepa.m@okicici",
            },
            "ORD-11006": {
                "id": "ORD-11006",
                "product": "Banarasi saree",
                "amount": 3499,
                "status": "returned",
                "upi": "rekha.s@okaxis",
            },
            "ORD-11007": {
                "id": "ORD-11007",
                "product": "Headphones",
                "amount": 1999,
                "status": "delivered",
                "upi": "arun.p@oksbi",
            },
            "ORD-11008": {
                "id": "ORD-11008",
                "product": "Organic mangoes",
                "amount": 599,
                "status": "delivered",
                "upi": "fatima.k@paytm",
            },
            "ORD-11009": {
                "id": "ORD-11009",
                "product": "Prescribed medicine",
                "amount": 450,
                "status": "delivered",
                "upi": "vijay.r@okhdfc",
            },
            # delivery scenarios
            "ORD-22001": {
                "id": "ORD-22001",
                "product": "Gaming console",
                "amount": 29999,
                "status": "lost",
                "upi": None,
            },
            "ORD-22002": {
                "id": "ORD-22002",
                "product": "Shoes + belt combo",
                "amount": 4499,
                "status": "delivered",
                "upi": None,
            },
            "ORD-22003": {
                "id": "ORD-22003",
                "product": "Silk saree",
                "amount": 5999,
                "status": "delivered",
                "upi": None,
            },
            "ORD-22004": {
                "id": "ORD-22004",
                "product": "Smartphone",
                "amount": 19999,
                "status": "delivered",
                "upi": None,
            },
            "ORD-22005": {
                "id": "ORD-22005",
                "product": "Laptop bag",
                "amount": 1499,
                "status": "in_transit",
                "upi": None,
            },
            "ORD-22006": {
                "id": "ORD-22006",
                "product": "Crockery set",
                "amount": 2199,
                "status": "in_transit",
                "upi": None,
            },
            "ORD-22007": {
                "id": "ORD-22007",
                "product": "Winter jacket",
                "amount": 2799,
                "status": "in_transit",
                "upi": None,
            },
            "ORD-22008": {
                "id": "ORD-22008",
                "product": "Kurta",
                "amount": 1199,
                "status": "in_transit",
                "upi": None,
            },
            "ORD-22009": {
                "id": "ORD-22009",
                "product": "Grocery order",
                "amount": 899,
                "status": "in_transit",
                "upi": None,
            },
            "ORD-22010": {
                "id": "ORD-22010",
                "product": "Urgent medicines",
                "amount": 799,
                "status": "in_transit",
                "upi": None,
            },
        }
        self.refunds: list[dict] = []
        self.emis: dict[str, dict] = {
            "LN-4471032": {
                "loan_id": "LN-4471032",
                "amount": 4250,
                "due_day": 5,
                "status": "active",
            },
            "LN-5000001": {
                "loan_id": "LN-5000001",
                "amount": 3200,
                "due_day": 3,
                "status": "active",
            },
            "LN-5000002": {
                "loan_id": "LN-5000002",
                "amount": 7500,
                "due_day": 1,
                "status": "active",
            },
            "LN-5000003": {
                "loan_id": "LN-5000003",
                "amount": 5000,
                "due_day": 10,
                "status": "overdue",
            },
            "LN-5000004": {
                "loan_id": "LN-5000004",
                "amount": 2800,
                "due_day": 15,
                "status": "bounced",
            },
            "LN-5000005": {
                "loan_id": "LN-5000005",
                "amount": 12000,
                "due_day": 5,
                "status": "active",
            },
            "LN-5000006": {
                "loan_id": "LN-5000006",
                "amount": 8500,
                "due_day": 7,
                "status": "active",
            },
            "LN-5000007": {
                "loan_id": "LN-5000007",
                "amount": 6000,
                "due_day": 8,
                "status": "active",
            },
            "LN-5000008": {
                "loan_id": "LN-5000008",
                "amount": 4500,
                "due_day": 20,
                "status": "active",
            },
            "LN-5000009": {
                "loan_id": "LN-5000009",
                "amount": 9000,
                "due_day": 12,
                "status": "disputed",
            },
        }
        self.deliveries: dict[str, dict] = {
            "ORD-77301": {
                "order_id": "ORD-77301",
                "status": "in_transit",
                "eta": "2026-10-07",
                "carrier": "Delhivery",
            },
            "ORD-22001": {
                "order_id": "ORD-22001",
                "status": "lost",
                "eta": None,
                "carrier": "FedEx",
            },
            "ORD-22002": {
                "order_id": "ORD-22002",
                "status": "delivered",
                "eta": "2026-09-30",
                "carrier": "BlueDart",
            },
            "ORD-22003": {
                "order_id": "ORD-22003",
                "status": "delivered",
                "eta": "2026-09-28",
                "carrier": "DTDC",
            },
            "ORD-22004": {
                "order_id": "ORD-22004",
                "status": "delivered",
                "eta": "2026-09-25",
                "carrier": "Delhivery",
            },
            "ORD-22005": {
                "order_id": "ORD-22005",
                "status": "in_transit",
                "eta": "2026-09-12",
                "carrier": "Delhivery",
            },
            "ORD-22006": {
                "order_id": "ORD-22006",
                "status": "in_transit",
                "eta": "2026-10-05",
                "carrier": "Ekart",
            },
            "ORD-22007": {
                "order_id": "ORD-22007",
                "status": "out_for_delivery",
                "eta": "2026-10-03",
                "carrier": "Shadowfax",
            },
            "ORD-22008": {
                "order_id": "ORD-22008",
                "status": "in_transit",
                "eta": "2026-10-06",
                "carrier": "DTDC",
            },
            "ORD-22009": {
                "order_id": "ORD-22009",
                "status": "in_transit",
                "eta": "2026-10-04",
                "carrier": "Delhivery",
            },
            "ORD-22010": {
                "order_id": "ORD-22010",
                "status": "in_transit",
                "eta": "2026-10-08",
                "carrier": "BlueDart",
            },
        }
        self.escalations: list[dict] = []

    def dump(self) -> dict:
        """Snapshot for RunRecord.final_db — plain dicts, no live references."""
        return {
            "orders": list(self.orders.values()),
            "refunds": list(self.refunds),
            "emis": list(self.emis.values()),
            "deliveries": list(self.deliveries.values()),
            "escalations": list(self.escalations),
        }


# ------------------------------------------------------------------ #
# Tool implementations (pure functions that mutate db)                 #
# ------------------------------------------------------------------ #


def lookup_order(db: MockDB, order_id: str) -> dict:
    row = db.orders.get(order_id)
    return row if row else {"error": f"Order {order_id!r} not found."}


def initiate_refund(db: MockDB, order_id: str, amount: int) -> dict:
    record: dict = {"order_id": order_id, "amount": amount, "status": "initiated"}
    db.refunds.append(record)
    return record


def check_emi_status(db: MockDB, loan_id: str) -> dict:
    row = db.emis.get(loan_id)
    return row if row else {"error": f"Loan {loan_id!r} not found."}


def reschedule_emi(db: MockDB, loan_id: str, new_due_day: int) -> dict:
    if loan_id not in db.emis:
        return {"error": f"Loan {loan_id!r} not found."}
    db.emis[loan_id]["due_day"] = new_due_day
    return {"loan_id": loan_id, "new_due_day": new_due_day, "status": "updated"}


def track_delivery(db: MockDB, order_id: str) -> dict:
    row = db.deliveries.get(order_id)
    return row if row else {"error": f"No delivery tracking for {order_id!r}."}


def escalate_to_human(db: MockDB, reason: str = "") -> dict:
    record: dict = {"reason": reason, "status": "escalated"}
    db.escalations.append(record)
    return record


# Maps tool name → (function, parameter names).
_DISPATCH: dict[str, tuple] = {
    "lookup_order": (lookup_order, ["order_id"]),
    "initiate_refund": (initiate_refund, ["order_id", "amount"]),
    "check_emi_status": (check_emi_status, ["loan_id"]),
    "reschedule_emi": (reschedule_emi, ["loan_id", "new_due_day"]),
    "track_delivery": (track_delivery, ["order_id"]),
    "escalate_to_human": (escalate_to_human, ["reason"]),
}


def _run_tool(db: MockDB, name: str, args: dict) -> dict:
    entry = _DISPATCH.get(name)
    if entry is None:
        return {"error": f"Unknown tool {name!r}."}
    fn, _ = entry
    try:
        return fn(db, **args)
    except TypeError as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool definitions (sent to the LLM as context)                        #
# ------------------------------------------------------------------ #

TOOL_DEFS: list[ToolDef] = [
    ToolDef(
        name="lookup_order",
        description="Look up an order by its ID and return its details.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    ),
    ToolDef(
        name="initiate_refund",
        description="Initiate a refund for an order. Call lookup_order first.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "integer"},
            },
            "required": ["order_id", "amount"],
        },
    ),
    ToolDef(
        name="check_emi_status",
        description="Check the current EMI status for a loan.",
        parameters={
            "type": "object",
            "properties": {"loan_id": {"type": "string"}},
            "required": ["loan_id"],
        },
    ),
    ToolDef(
        name="reschedule_emi",
        description="Change the monthly due date for a loan's EMI.",
        parameters={
            "type": "object",
            "properties": {
                "loan_id": {"type": "string"},
                "new_due_day": {"type": "integer", "minimum": 1, "maximum": 28},
            },
            "required": ["loan_id", "new_due_day"],
        },
    ),
    ToolDef(
        name="track_delivery",
        description="Get the delivery status and ETA for an order.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    ),
    ToolDef(
        name="escalate_to_human",
        description="Transfer the caller to a human agent with an optional reason.",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    ),
]


# ------------------------------------------------------------------ #
# Reference agent                                                      #
# ------------------------------------------------------------------ #


def _to_openai_tool(t: ToolDef) -> dict:
    """Convert a ToolDef into the OpenAI function-tool schema the API expects."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


class ReferenceAgent:
    """An agent backed by an LLM Role and a MockDB.

    respond() runs a tool-use loop: it offers the tool schemas to the model, and
    whenever the model requests tool calls it executes them against self.db, feeds
    the results back as tool messages, and calls the model again. The loop ends
    when the model replies with plain text (or MAX_TOOL_ROUNDS is hit). This lets
    the model see tool results and act on them — e.g. refund the amount it looked
    up — instead of narrating actions it never took. No special-casing for
    MockProvider: the same path runs for mock and real providers.
    """

    def __init__(self, role: Role, db: MockDB | None = None) -> None:
        self._role = role
        self.db = db if db is not None else MockDB()

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn:
        messages = [ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT)]
        for turn in turns:
            role = "user" if turn.speaker == "caller" else "assistant"
            messages.append(ChatMessage(role=role, content=turn.text))

        tool_schemas = [_to_openai_tool(t) for t in tools]
        executed: list[ToolCall] = []
        prompt_tokens = completion_tokens = 0
        latency = 0.0
        text = ""
        finish_reason: str | None = None

        for _ in range(MAX_TOOL_ROUNDS):
            resp = await self._role.chat(messages, temperature=0.0, tools=tool_schemas)
            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            latency += resp.latency_s
            text = resp.text
            finish_reason = resp.finish_reason

            if not resp.tool_calls:
                break

            # Give each requested call a stable id, shared between the assistant
            # tool-call message and its matching tool-result message.
            ids = [tc.id or f"call_{i}" for i, tc in enumerate(resp.tool_calls)]
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=resp.text,
                    tool_calls=[
                        {
                            "id": ids[i],
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                            # Echo provider passthrough (e.g. Gemini thought_signature).
                            **(tc.extra or {}),
                        }
                        for i, tc in enumerate(resp.tool_calls)
                    ],
                )
            )
            for i, tc in enumerate(resp.tool_calls):
                result = _run_tool(self.db, tc.name, tc.args)  # side-effect on self.db
                executed.append(ToolCall(name=tc.name, args=tc.args))
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=ids[i],
                        content=json.dumps(result),
                    )
                )

        return AgentTurn(
            text=text,
            tool_calls=executed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            finish_reason=finish_reason,
        )
