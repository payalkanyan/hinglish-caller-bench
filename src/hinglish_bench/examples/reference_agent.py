"""Reference agent backed by an in-memory MockDB.

The six mock tools (lookup_order, initiate_refund, check_emi_status,
reschedule_emi, track_delivery, escalate_to_human) are pure functions that
mutate a MockDB instance passed in at construction time.

ReferenceAgent.respond() calls its Role (which may be MockProvider or a real
LLM) to decide which tools to call, then executes them. There is no special-
casing for MockProvider — the agent always follows the same code path.
"""

from __future__ import annotations

from hinglish_bench.agent import AgentTurn, ToolDef
from hinglish_bench.providers.base import ChatMessage, Role
from hinglish_bench.schemas import ToolCall, Turn

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
            "ORD-88412": {
                "id": "ORD-88412",
                "product": "Floral kurta",
                "amount": 1499,
                "status": "delivered",
                "customer": "Meera Iyer",
                "city": "Bengaluru",
                "upi": "meera.iyer@okaxis",
            },
            "ORD-77301": {
                "id": "ORD-77301",
                "product": "Men's jeans",
                "amount": 899,
                "status": "in_transit",
                "customer": "Arjun Sharma",
                "city": "Delhi",
                "upi": None,
            },
        }
        self.refunds: list[dict] = []
        self.emis: dict[str, dict] = {
            "LN-4471032": {
                "loan_id": "LN-4471032",
                "borrower": "Ramesh Gupta",
                "amount": 4250,
                "due_day": 5,
                "status": "active",
            }
        }
        self.deliveries: dict[str, dict] = {
            "ORD-77301": {
                "order_id": "ORD-77301",
                "status": "in_transit",
                "eta": "2026-10-07",
                "carrier": "Delhivery",
            }
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


class ReferenceAgent:
    """An agent backed by an LLM Role and a MockDB.

    respond() makes exactly one LLM call. The LLM (or mock) decides which
    tools to call via the tool_calls field of the ChatResponse. The agent
    then executes those calls against self.db and returns the results.
    There is no special-casing for MockProvider.
    """

    def __init__(self, role: Role, db: MockDB | None = None) -> None:
        self._role = role
        self.db = db if db is not None else MockDB()

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn:
        messages = [ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT)]
        for turn in turns:
            role = "user" if turn.speaker == "caller" else "assistant"
            messages.append(ChatMessage(role=role, content=turn.text))

        resp = await self._role.chat(messages, temperature=0.0)

        # Execute each tool call the LLM requested, in order.
        executed: list[ToolCall] = []
        for tc in resp.tool_calls:
            _run_tool(self.db, tc.name, tc.args)  # side-effect on self.db
            executed.append(ToolCall(name=tc.name, args=tc.args))

        return AgentTurn(
            text=resp.text,
            tool_calls=executed,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_s=resp.latency_s,
        )
