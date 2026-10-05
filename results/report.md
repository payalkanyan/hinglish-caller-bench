# hinglish-caller-bench Report
**Runs:** 90  **Overall success:** 79/90
## Results by persona × domain
| Persona | Domain | N | Success% (95% CI) | Tool% (95% CI) | pass@k | Lang-fit |
|---|---|---|---|---|---|---|
| english | delivery | 9 | 88.9% [0.56, 0.98] | 44.4% [0.19, 0.73] | 0.667 | 5.0 |
| english | emi_reminder | 9 | 100.0% [0.70, 1.00] | 33.3% [0.12, 0.65] | 1.000 | 5.0 |
| hindi_devanagari | delivery | 9 | 55.6% [0.27, 0.81] | 33.3% [0.12, 0.65] | 0.333 | 5.0 |
| hindi_devanagari | emi_reminder | 9 | 100.0% [0.70, 1.00] | 88.9% [0.56, 0.98] | 1.000 | 5.0 |
| hindi_roman | delivery | 9 | 77.8% [0.45, 0.94] | 55.6% [0.27, 0.81] | 0.333 | 5.0 |
| hindi_roman | emi_reminder | 9 | 100.0% [0.70, 1.00] | 77.8% [0.45, 0.94] | 1.000 | 5.0 |
| hinglish | delivery | 9 | 88.9% [0.56, 0.98] | 0.0% [0.00, 0.30] | 0.667 | 5.0 |
| hinglish | emi_reminder | 9 | 100.0% [0.70, 1.00] | 44.4% [0.19, 0.73] | 1.000 | 5.0 |
| switcher | delivery | 9 | 66.7% [0.35, 0.88] | 22.2% [0.06, 0.55] | 0.333 | 5.0 |
| switcher | emi_reminder | 9 | 100.0% [0.70, 1.00] | 22.2% [0.06, 0.55] | 1.000 | 5.0 |

## Failure examples (3 worst transcripts)
### 1. `delivery_damaged_carrier` / `hindi_devanagari` / run 1
**End reason:** caller_ended  
**Task completion:** FAIL  
**Missing tools:** track_delivery, escalate_to_human  

> **Caller:** नमस्ते, मैंने हाल ही में एक सरोकरी सेट का ऑर्डर किया था। डिलीवरी के समय जिस बॉक्स को मुझे दिया गया था, वो बहुत ज्यादा कुचला हुआ था और अंदर से टूटे हुए कांच की आवाज आ रही थी। मैंने खोलने से पहले ही फोटो ले ली थी। मुझे इसकी शिकायत दर्ज करानी है और रिप्लेसमेंट या रिफंड चाहिए।
>
> **Agent:**  नमस्ते जी। आपके इस अनुभव के लिए मुझे खेद है। 

इस समस्या का समाधान करने के लिए मुझे आपके ऑर्डर नंबर की आवश्यकता होगी। कृपया अपना ऑर्डर नंबर बता दीजिए, ताकि मैं आगे की कार्रवाई कर सकूं।
>
> **Caller:** जी हाँ, मेरा ऑर्डर आईडी ORD-22006 है।
>
> **Agent:**  आपके ऑर्डर (ORD-22006) के लिए ₹2,199 का रिफंड सफलतापूर्वक शुरू कर दिया गया है। 

चूंकि बॉक्स क्षतिग्रस्त था, आपने फोटो ले ली है यह बहुत अच्छा किया। रिफंड की राशि कुछ ही दिनों में आपके ओरिजिनल पेमेंट मेथड पर पहुँच जाएगी। क्या मैं किसी और सहायता में आपकी मदद कर सकता हूँ?
>
> **Caller:** जी, आपने सही किया। मैंने तो शुरू में रिप्लेसमेंट का भी विकल्प पूछा था, क्योंकि मुझे वो सरोकरी सेट चाहिए था। अगर रिफंड ही हो रहा है, तो ठीक है, लेकिन मुझे बताना होगा कि पैसा वापस कितने दिनों में मिलेगा? और क्या इससे पहले मुझे कोई कॉन्फर्मेशन मिलेगा?
>
> **Agent:**  रिफंड की राशि सामान्यतः 3 से 5 कार्य दिवसों (business days) के भीतर आपके ओरिजिनल पेमेंट मेथड में वापस आ जाएगी। 

जैसे ही रिफंड प्रोसेस पूरा होगा, आपको आपके रजिस्टर्ड मोबाइल नंबर और ईमेल पर एक कन्फर्मेशन मैसेज (पुष्टि संदेश) मिल जाएगा। 

अगर आप चाहें तो उसी सरोकरी सेट के लिए नया ऑर्डर भी कर सकती हैं, क्योंकि यह आइटम अभी उपलब्ध है। क्या मैं इसमें आपकी कोई और सहायता कर सकता हूँ?
>
> *(transcript trimmed to 6 turns)*

### 2. `delivery_damaged_carrier` / `hindi_devanagari` / run 0
**End reason:** max_turns  
**Task completion:** FAIL  
**Missing tools:** track_delivery, escalate_to_human  

> **Caller:** नमस्ते, मुझे अपने ऑर्डर के बारे में शिकायत दर्ज करानी है।
>
> **Agent:**  नमस्ते! मैं आपकी शिकायत दर्ज करने में मदद कर सकता हूँ। कृपया मुझे अपना **ऑर्डर आईडी (Order ID)** बताएँ ताकि मैं स्थिति की जाँच कर सकूँ।
>
> **Caller:** जी हाँ, मेरा ऑर्डर आईडी है ORD-22006।
>
> **Agent:**  आपका ऑर्डर (ORD-22006 - क्रॉकरी सेट) अभी रास्ते में है (In Transit) और इसकी कुल राशि ₹2,199 है। 

आप इस ऑर्डर के बारे में क्या करना चाहते हैं? क्या आप इसकी डिलीवरी ट्रैक करना चाहते हैं या इसके लिए रिफंड शुरू करना चाहते हैं?
>
> **Caller:** नहीं, मैं ट्रैक नहीं करना चाहता। मुझे पता है कि यह डिलीवर हो चुका है।

समस्या यह है कि जब डिलीवरी वाले ने मुझे पैकेज दिया, तो उसका बॉक्स पूरी तरह से कुचल दिया हुआ था (crushed)। मैंने खोलने से पहले ही अंदर से टूटे हुए टुकड़ों की आवाज़ सुनी थी और उसका फोटो भी ले लिया था।

मैं चाहता हूँ कि इसकी शिकायत दर्ज की जाए और मुझे या तो रिफंड मिले या फिर रिप्लेसमेंट का ऑर्डर दिया जाए।
>
> **Agent:**  आपके ऑर्डर (ORD-22006) के लिए ₹2,199 का रिफंड सफलतापूर्वक शुरू कर दिया गया है। 

यह रिफंड आम तौर पर 3-5 कार्य दिवसों (business days) में आपके मूल भुगतान माध्यम में जमा हो जाएगा। टूटे हुए सामान के लिए हुई असुविधा के लिए हमें खेद है। क्या मैं किसी और सहायता में आपकी मदद कर सकता हूँ?
>
> *(transcript trimmed to 6 turns)*

### 3. `delivery_damaged_carrier` / `hindi_roman` / run 2
**End reason:** max_turns  
**Task completion:** FAIL  
**Missing tools:** track_delivery, escalate_to_human  

> **Caller:** Namaste, mera order toh aaya hai par box itna chaka hua tha ki andar se tootne ki awaaz aa rahi thi. Maine kholne se pehle photo le li thi. Abhi check karo ki yeh kya hua?
>
> **Agent:**  Namaste! Yeh sunkar mujhe bahut dukh hua ki aapka order damaged condition mein mila hai. Aapne sahi kiya ki kholne se pehle photo le li. 

Kripya aap mujhe apna **order ID** bata sakte hain, taaki main turant check kar sakoon ki aage kya karna hai?
>
> **Caller:** Order ID hai ORD-22006.
>
> **Agent:**  Maine aapka order check kar liya hai (Order ID: ORD-22006, Crockery set). Yeh order abhi transit mein hai ya deliver ho chuka hai? Kyunki system mein status "in_transit" dikha raha hai, lekin aapne bataya ki box aa gaya hai. 

Kya main aapke is damaged order ke liye refund initiate kar doon? Refund amount ₹2,199 hai.
>
> **Caller:** Bhai, box toh deliver ho chuka hai, maine photo bhi le li thi. Ab refund nahi chahiye, replacement ship karo.
>
> **Agent:**  Main samajh sakta hoon ki aapko replacement chahiye. Lekin hamare system mein abhi direct replacement ka option nahi hai. Iske liye hume pehle refund initiate karna hoga aur phir aap naya order place kar sakte hain. 

Kya main aapke order (ORD-22006) ke liye ₹2,199 ka refund initiate kar doon?
>
> *(transcript trimmed to 6 turns)*

