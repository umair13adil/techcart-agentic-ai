"""Training data for the query-intent classifier.

Train: template-generated utterances (cheap, controllable, covers the label space).
Test : hand-written queries with different phrasing, typos and code-switching, so the reported
       accuracy is not just "the model memorised my templates". In production this test set grows
       from labelled search logs (active learning on low-confidence queries).
"""
from __future__ import annotations

import random

INTENTS = ["product_search", "comparison", "recommendation", "order_status", "return_refund",
           "policy_question", "purchase", "quote_request", "other"]

_ITEMS = ["laptop", "gaming laptop", "notebook", "macbook", "ultrabook", "monitor", "4k monitor", "gaming monitor",
          "keyboard", "mechanical keyboard", "mouse", "wireless mouse", "headset", "headphones", "webcam",
          "usb-c hub", "docking station", "ssd", "external hard drive", "ram", "graphics card", "rtx 4070",
          "router", "mesh wifi", "laptop bag", "backpack", "charger", "power bank", "cooling pad", "mini pc",
          "thinkpad", "dell xps", "rog zephyrus", "mx master", "2-in-1 laptop", "business laptop"]
_USES = ["gaming", "programming", "video editing", "university", "office work", "travel", "graphic design",
         "streaming", "machine learning", "my kids", "zoom calls", "coding"]
_BUDGET = ["under 3000 SAR", "below 5k", "less than 4000 riyal", "around 2500", "max 6000", "between 3000 and 4500",
           "on a budget", "cheap", "under 200", "up to 10,000 SAR", ""]
_ORDER = ["O0012345", "O0008891", "my order", "order number O0101010", "the package", "my delivery", "my parcel"]

TEMPLATES: dict[str, list[str]] = {
    "product_search": [
        "show me {item} {budget}", "i need a {item} for {use} {budget}", "looking for a {item}", "{item} {budget}",
        "do you have {item} in stock", "best {item} for {use}", "find me a {item} with 16gb ram",
        "any {item} with 1tb ssd {budget}", "{item} for {use}", "what {item} do you have for {use}",
        "i want something for {use} {budget}", "light {item} for {use}", "{item} with good battery life",
        "search {item}", "need a new {item} {budget}", "available {item} options", "cheapest {item}",
        "{item} 14 inch {budget}", "is there a {item} with rtx 4060", "a {item} that can run {use}",
        "got any {item}", "{item} w/ long battery {budget}", "something to use for {use}", "quiet {item} for {use}",
        "recommend a {item} for {use} {budget}", "{item} for my {use}, {budget}",
    ],
    "comparison": [
        "compare {item} and {item2}", "{item} vs {item2}", "which is better {item} or {item2}",
        "difference between {item} and {item2}", "should i get {item} or {item2} for {use}",
        "{item} or {item2} which one", "how does {item} compare to {item2}", "pros and cons of {item} vs {item2}",
        "is {item} worth it over {item2}", "rtx 4060 vs rtx 4070 for {use}", "16gb vs 32gb ram for {use}",
    ],
    "recommendation": [
        "what accessories go with my {item}", "what else should i buy with a {item}", "recommend something for me",
        "what goes well with {item}", "suggest accessories for {use}", "what should i buy next",
        "complete my setup for {use}", "i just bought a {item} what do i need", "you might also like",
        "recommend a bundle for {use}", "any suggestions based on my last order", "what pairs with a {item}",
        "upgrade ideas for my {item}", "gift ideas for a gamer", "what do other customers buy with {item}",
    ],
    "order_status": [
        "where is {order}", "track {order}", "has {order} shipped", "when will {order} arrive",
        "status of {order}", "{order} still not delivered", "delivery update for {order}",
        "my order hasnt arrived yet", "tracking number for {order}", "is {order} out for delivery",
        "check order {order}", "how long until {order} gets here",
    ],
    "return_refund": [
        "i want to return my {item}", "how do i return {order}", "refund for {order}", "my {item} is broken",
        "{item} arrived damaged", "can i exchange my {item}", "return request {order}", "where is my refund",
        "the {item} is not working i want my money back", "wrong item delivered in {order}", "cancel {order}",
        "i received a defective {item}", "start a return", "exchange {item} for a different color",
        "{item} stopped working after {n} days", "you sent the wrong {item}", "can i send the {item} back",
        "refund not received yet", "dead pixels on the new {item}", "the {item} is faulty", "changed my mind about the {item}",
        "{item} keeps disconnecting, want a replacement", "box was opened and {item} scratched",
    ],
    "policy_question": [
        "do you offer installments", "can i pay with tabby", "is cash on delivery available",
        "what is the warranty on {item}", "how much is shipping to jeddah", "do you ship outside saudi",
        "what is your return policy", "do you price match", "how long is the warranty", "do you accept apple pay",
        "is vat included in the price", "do you have a trade in program", "how do you use my data",
        "what is techcare plus", "delivery time to dammam", "do you have same day delivery",
        "can i get a vat invoice", "what payment methods do you accept", "does warranty cover water damage",
        "can i split the payment", "installment plans available?", "is cod possible for a {n}00 sar order",
        "what is the cod fee", "does the protection plan cover accidental drops", "how many days to ship to {city}",
        "do you share my personal information", "are prices inclusive of vat", "what does the warranty cover",
        "is there a fee for returns", "how long does delivery take to {city}", "pay later options",
    ],
    "purchase": [
        "buy the {item}", "add {item} to cart", "i'll take the {item}", "checkout", "place the order",
        "purchase {item} now", "order the {item} for me", "add it to my cart", "proceed to payment",
        "i want to buy this", "complete my purchase", "yes order it", "buy 2 {item}", "pay with mada",
        "confirm the order", "get me the {item} delivered to riyadh",
    ],
    "quote_request": [
        "i need a quotation for {n} {item}", "quote for {n} laptops for my company", "bulk order of {n} {item}",
        "corporate pricing for {n} {item}", "can you send a quote for our office {n} units",
        "rfq {n} {item} for school", "price for {n} {item} with autopilot", "we need {n} {item} for new hires",
        "b2b discount for {n} {item}", "request quotation {n} monitors and docks",
    ],
    "other": [
        "hello", "hi there", "who are you", "thanks", "tell me a joke", "what is the weather", "good morning",
        "are you a bot", "bye", "ok", "how are you", "what can you do", "help", "salam", "nice", "cool thanks",
        "who won the match yesterday", "write me a poem", "what time is it",
    ],
}

TEST_SET: list[tuple[str, str]] = [
    ("Need a lappy for my CS degree, nothing over 4k", "product_search"),
    ("got any curved screens for ps5 and pc", "product_search"),
    ("lightweight notebook w/ long battery, mostly excel", "product_search"),
    ("rtx laptop 5000 riyal budget", "product_search"),
    ("something to edit 4k youtube videos on", "product_search"),
    ("silent mouse for the library", "product_search"),
    ("portable ssd for my footage 2tb", "product_search"),
    ("ابغى لابتوب قيمنق", "product_search"),
    ("macbook air or zenbook 14 for uni?", "comparison"),
    ("xps 15 versus proart p16 for premiere", "comparison"),
    ("is the oled monitor really better than the ips one", "comparison"),
    ("4060 vs 4070 laptop worth the extra 1500?", "comparison"),
    ("thinkpad t14 or elitebook 840 which lasts longer", "comparison"),
    ("i bought the legion last month what else would be useful", "recommendation"),
    ("finish off my desk setup please", "recommendation"),
    ("what do ppl usually pair with the mx keys", "recommendation"),
    ("anything I'm missing for streaming?", "recommendation"),
    ("what would you suggest i get next", "recommendation"),
    ("wheres my stuff, ordered O0011122 on sunday", "order_status"),
    ("parcel still not here!!", "order_status"),
    ("has my monitor left the warehouse", "order_status"),
    ("ETA for order O0099887?", "order_status"),
    ("tracking link pls", "order_status"),
    ("headset mic stopped working after 3 days", "return_refund"),
    ("you sent me the wrong ssd", "return_refund"),
    ("i changed my mind can i send the laptop back", "return_refund"),
    ("money still not back in my account from the return", "return_refund"),
    ("screen has dead pixels out of the box", "return_refund"),
    ("can I split the payment into 4", "policy_question"),
    ("cod allowed for 3000 sar order?", "policy_question"),
    ("does the 2 year plan cover if I drop it", "policy_question"),
    ("how many days for delivery to abha", "policy_question"),
    ("is my personal data shared with third parties", "policy_question"),
    ("do prices include vat", "policy_question"),
    ("ok add the g502 to my basket", "purchase"),
    ("go ahead and check out with apple pay", "purchase"),
    ("yep buy that one", "purchase"),
    ("I'd like to order two of the keychron keyboards", "purchase"),
    ("place it, ship to my jeddah address", "purchase"),
    ("our ministry needs 40 latitudes, send pricing", "quote_request"),
    ("formal quote for 25 thinkpads + docks for our clinic", "quote_request"),
    ("bulk pricing on 100 monitors?", "quote_request"),
    ("school lab needs 30 desktops, can you quote", "quote_request"),
    ("hey", "other"),
    ("thank you so much", "other"),
    ("what's the capital of france", "other"),
    ("are you chatgpt", "other"),
]


def generate(n_per_intent: int = 260, seed: int = 11) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    data = []
    for intent, temps in TEMPLATES.items():
        for _ in range(n_per_intent):
            t = rng.choice(temps)
            item, item2 = rng.sample(_ITEMS, 2)
            s = t.format(item=item, item2=item2, use=rng.choice(_USES), budget=rng.choice(_BUDGET),
                         order=rng.choice(_ORDER), n=rng.choice([10, 15, 20, 25, 30, 50, 100]),
                         city=rng.choice(["jeddah", "dammam", "abha", "tabuk", "riyadh", "mecca"]))
            s = " ".join(s.split())
            # light noise: casing, punctuation, dropped letters
            r = rng.random()
            if r < 0.15:
                s = s.capitalize() + rng.choice(["?", "!", "."])
            elif r < 0.25 and len(s) > 8:
                i = rng.randrange(1, len(s) - 1)
                s = s[:i] + s[i + 1:]
            data.append((s, intent))
    rng.shuffle(data)
    return data
