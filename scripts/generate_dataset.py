"""Seeded generator for the synthetic support-ticket dataset.

Every ticket is assembled from hand-written templates and slot fillers, then
roughed up with noise so the classification task is not trivially separable:

* typos (swapped / dropped / doubled characters) and casing changes
* "ambiguous" tickets that also contain a sentence from a second category
* vague tickets with almost no category signal
* priority cues that are missing or point at a neighbouring priority level
* label noise: a small share of category and priority labels are flipped

The same seed always produces a byte-identical CSV.

Usage:
    python -m scripts.generate_dataset                 # data/tickets.csv, 3000 rows, seed 42
    python -m scripts.generate_dataset --n 500 --seed 7 --out data/small.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CATEGORIES = [
    "billing",
    "login_access",
    "bug",
    "feature_request",
    "shipping",
    "account",
    "refund_return",
]
PRIORITIES = ["low", "medium", "high", "urgent"]

# Class imbalance on purpose: real queues are never uniform.
CATEGORY_WEIGHTS = {
    "billing": 0.18,
    "login_access": 0.17,
    "bug": 0.19,
    "feature_request": 0.10,
    "shipping": 0.14,
    "account": 0.11,
    "refund_return": 0.11,
}

# P(priority | category). Feature requests skew low, login problems skew high.
PRIORITY_GIVEN_CATEGORY = {
    "billing": [0.20, 0.40, 0.30, 0.10],
    "login_access": [0.10, 0.30, 0.35, 0.25],
    "bug": [0.15, 0.35, 0.30, 0.20],
    "feature_request": [0.60, 0.30, 0.08, 0.02],
    "shipping": [0.20, 0.40, 0.30, 0.10],
    "account": [0.35, 0.40, 0.20, 0.05],
    "refund_return": [0.20, 0.45, 0.28, 0.07],
}

AMBIGUOUS_RATE = 0.12
VAGUE_RATE = 0.04
CATEGORY_LABEL_NOISE = 0.04
PRIORITY_LABEL_NOISE = 0.04
TYPO_TICKET_RATE = 0.35
PRIORITY_CUE_RATE = 0.78  # share of tickets whose body carries any urgency cue
WRONG_CUE_RATE = 0.12  # of those, share whose cue belongs to a neighbouring level

PRODUCTS = [
    "the dashboard", "the mobile app", "the web app", "the reports page", "the checkout page",
    "the admin console", "the Android app", "the iOS app", "the export tool", "the API",
    "the settings page", "the calendar view", "the search bar", "the invoice screen",
]
PLANS = ["Starter", "Pro", "Team", "Business", "annual", "monthly"]
AMOUNTS = ["$9.99", "$19", "$29.00", "$49", "$99.00", "$120", "$249", "$14.50"]
ITEMS = [
    "headphones", "a desk lamp", "a keyboard", "two phone cases", "a backpack", "a monitor stand",
    "a coffee grinder", "running shoes", "a jacket", "a replacement charger", "a water bottle",
]
CARRIERS = ["the courier", "the carrier", "the delivery company", "the postal service"]
DAYS = ["2 days", "3 days", "a week", "5 days", "ten days", "two weeks", "yesterday", "Monday"]
ORDER_IDS = ["#48213", "#50977", "#61204", "#39018", "#72551", "#80463", "#11890", "#93377"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"]
FEATURES = [
    "dark mode", "CSV export", "a Slack integration", "bulk editing", "two-way calendar sync",
    "custom fields", "keyboard shortcuts", "an offline mode", "scheduled reports", "SSO support",
    "a public API endpoint for tags", "saved filters",
]

SUBJECTS = {
    "billing": [
        "Charged twice this month", "Question about my invoice", "Wrong amount on invoice",
        "Unexpected charge of {amount}", "Billing issue on {plan} plan", "Card declined on renewal",
        "Need a copy of my invoice", "Why was I charged {amount}?", "Payment failed", "VAT missing on invoice",
        "Subscription renewed without notice", "Invoice shows the wrong plan",
    ],
    "login_access": [
        "Can't log in", "Password reset not working", "Locked out of my account", "2FA code never arrives",
        "Login keeps failing", "Reset link expired", "Unable to sign in on {product}", "Access denied after login",
        "SSO login loop", "Verification email not received", "Stuck on the sign in screen",
    ],
    "bug": [
        "{product} crashes on load", "Error 500 when saving", "Button does nothing", "Data not syncing",
        "Page is blank in {browser}", "Export produces an empty file", "Something broke after the update",
        "Search returns wrong results", "App freezes when uploading", "Chart shows wrong numbers",
        "Unexpected error message", "Notifications arrive twice",
    ],
    "feature_request": [
        "Feature request: {feature}", "Please add {feature}", "Suggestion for {product}", "Would love {feature}",
        "Idea: {feature}", "Any plans for {feature}?", "Request - {feature}", "Small improvement idea",
    ],
    "shipping": [
        "Where is my order {order}?", "Package has not arrived", "Tracking not updating", "Order {order} delayed",
        "Delivered to the wrong address", "Change delivery address", "Package marked delivered but missing",
        "Shipping status for {order}", "Parcel stuck in transit", "Late delivery",
    ],
    "account": [
        "Update my email address", "Change account owner", "How do I delete my account?", "Add a team member",
        "Change company name on profile", "Close my account", "Merge two accounts", "Update phone number",
        "Change my username", "Transfer workspace to a colleague", "Downgrade seats on the account",
    ],
    "refund_return": [
        "Refund request for order {order}", "Want to return {item}", "Refund not received", "Return label needed",
        "Item arrived damaged - refund?", "Cancel and refund my order", "How do I return this?",
        "Money back for {plan} plan", "Refund still pending", "Wrong item, want to send it back",
    ],
}

CORE = {
    "billing": [
        "I was charged {amount} twice on my card for the {plan} plan.",
        "My latest invoice shows {amount} but my plan should cost less than that.",
        "The payment for my subscription failed even though the card is valid.",
        "I need an invoice with our company tax number on it for accounting.",
        "My card was billed after I had already switched to the {plan} plan.",
        "There is a charge of {amount} on my statement that I do not recognise.",
        "The renewal went through at the old price instead of the discounted one.",
        "Can you explain the extra line item on this month's bill?",
        "The receipt I got by email has the wrong billing address.",
        "I upgraded the plan but the invoice still lists the cheaper tier and a proration I don't understand.",
    ],
    "login_access": [
        "I can't sign in, it says my password is incorrect even after I reset it.",
        "The password reset email never shows up in my inbox or spam.",
        "My account got locked after a few failed login attempts.",
        "The two-factor code from the authenticator app is rejected every time.",
        "After entering my credentials the page just reloads back to the login form.",
        "I get an access denied message when I try to open {product}.",
        "The reset link says it has expired as soon as I click it.",
        "Single sign-on sends me in a loop and I never reach the home screen.",
        "I changed my phone and now I cannot receive the verification code.",
        "It says my session expired right after I log in.",
    ],
    "bug": [
        "{product} crashes every time I open it in {browser}.",
        "When I click save I get an error 500 and my changes are lost.",
        "The export finishes but the downloaded file is empty.",
        "Data entered on my phone does not sync to the web version.",
        "The page stays blank and the console shows a script error.",
        "Since the last update the totals in the chart are clearly wrong.",
        "Uploading a file larger than a few MB makes the app freeze.",
        "The search box returns results that have nothing to do with what I typed.",
        "Clicking the submit button does nothing at all, no error, no loading spinner.",
        "I receive every notification twice and sometimes with the wrong title.",
    ],
    "feature_request": [
        "It would be great if you could add {feature} to {product}.",
        "Our team would really benefit from {feature}.",
        "Have you considered supporting {feature}? We currently use a workaround.",
        "I'd like to suggest {feature} as an improvement for a future release.",
        "Is {feature} on your roadmap? It is the main thing we are missing.",
        "A nice to have would be {feature}, especially for larger teams.",
        "Could you add an option for {feature}? Competitors already offer it.",
        "I love the product, one idea: {feature}.",
    ],
    "shipping": [
        "My order {order} was supposed to arrive {days} ago and it is still not here.",
        "The tracking page for {order} has not updated in {days}.",
        "The parcel is marked as delivered but nothing was left at my door.",
        "I ordered {item} and {carrier} says it is stuck at the depot.",
        "I need to change the delivery address for order {order} before it ships.",
        "The package was delivered to my old address by mistake.",
        "Can you tell me when {item} will be dispatched? I have no tracking number yet.",
        "{carrier} attempted delivery while nobody was home, how do I reschedule?",
        "Half of my order arrived, the box with {item} is missing.",
        "Shipping was quoted as {days} and it has been much longer.",
    ],
    "account": [
        "I need to change the email address on my account to a new one.",
        "Please transfer ownership of our workspace to my colleague.",
        "How can I permanently delete my account and all of my data?",
        "I want to add another team member but I can't find where to invite them.",
        "Our company was renamed, how do I update the name on our profile?",
        "I accidentally created two accounts and would like them merged.",
        "Please update the phone number linked to my profile.",
        "I would like to reduce the number of seats on our account.",
        "How do I change my display name and username?",
        "The previous admin left the company and we need a new account owner.",
    ],
    "refund_return": [
        "I would like a refund for order {order}, the item was not as described.",
        "I returned {item} {days} ago and the refund has not reached my card.",
        "{item} arrived damaged and I want to send it back.",
        "Please cancel my order and refund the {amount} I paid.",
        "I need a return label for {item}, it does not fit.",
        "I cancelled the {plan} plan within the trial window and want my money back.",
        "You sent the wrong item, how do I return it and get the right one?",
        "The refund was approved but I still have not received {amount}.",
        "What is your return policy for {item} bought {days} ago?",
        "I'd like to exchange {item} or get a refund if that's not possible.",
    ],
}

# Sentences with little or no category signal, used for vague tickets and filler.
VAGUE_SUBJECTS = ["Need help", "Question", "Issue", "Help please", "Problem with my order", "Not working",
                  "Quick question", "Following up", "Support needed"]
VAGUE_BODIES = [
    "Something is not right and I'm not sure who to ask.",
    "It is not working the way it did last week.",
    "I have a problem and need someone to look into it.",
    "I wrote before and did not hear back.",
    "Can somebody contact me about this?",
    "There is an issue with my last order, please check.",
    "I'm confused about what happened here.",
]

OPENERS = [
    "Hi,", "Hello,", "Hi team,", "Hey,", "Good morning,", "Hello support,", "To whom it may concern,", "", "", "",
]
DETAILS = [
    "I already tried logging out and back in.", "I have attached a screenshot.", "This started {days} ago.",
    "I'm using {browser} on a laptop.", "My order number is {order}.", "I have been a customer for two years.",
    "I checked the help centre and could not find an answer.", "A colleague has the same problem.",
    "I tried on my phone as well.", "I contacted you about this before.", "Nothing changed on my side.",
]
CLOSERS = ["Thanks.", "Thank you!", "Regards", "Thanks in advance.", "Cheers", "Please advise.",
           "Looking forward to your reply.", "", "", ""]

PRIORITY_CUES = {
    "low": [
        "No rush on this.", "Whenever you get a chance.", "Not urgent, just wanted to mention it.",
        "This is a minor thing.", "Low priority from my side.", "Just curious, no hurry.",
    ],
    "medium": [
        "It would be good to sort this out this week.", "Please look into it when you can.",
        "It is a bit annoying but I can work around it.", "I'd appreciate an update in the next few days.",
        "This is slowing me down somewhat.",
    ],
    "high": [
        "This is blocking my work today.", "I need this fixed soon, it affects several people on my team.",
        "Please prioritise this, we have a deadline tomorrow.", "This is causing real problems for us.",
        "I can't do my job until this is resolved.", "Second time I'm writing about this, please escalate.",
    ],
    "urgent": [
        "This is urgent, our whole team is locked out.", "URGENT - we are losing money every hour.",
        "Production is down for all of our customers.", "I need help immediately, this is critical.",
        "This is an emergency, please call me as soon as possible.", "ASAP please, the launch is in one hour.",
    ],
}
URGENT_SUBJECT_PREFIX = ["URGENT: ", "ASAP - ", "Critical: ", "[URGENT] "]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        plan=rng.choice(PLANS),
        amount=rng.choice(AMOUNTS),
        item=rng.choice(ITEMS),
        carrier=rng.choice(CARRIERS),
        days=rng.choice(DAYS),
        order=rng.choice(ORDER_IDS),
        browser=rng.choice(BROWSERS),
        feature=rng.choice(FEATURES),
    )


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _add_typos(text: str, rng: random.Random, rate: float = 0.03) -> str:
    """Swap, drop or double roughly `rate` of the letters."""
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "double"])
            if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.extend([chars[i + 1], c])
                i += 2
                continue
            if op == "drop":
                i += 1
                continue
            out.extend([c, c])
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _neighbour(priority: str, rng: random.Random) -> str:
    idx = PRIORITIES.index(priority)
    options = [j for j in (idx - 1, idx + 1) if 0 <= j < len(PRIORITIES)]
    return PRIORITIES[rng.choice(options)]


def make_ticket(ticket_id: int, rng: random.Random) -> dict[str, str | int]:
    category = rng.choices(CATEGORIES, weights=[CATEGORY_WEIGHTS[c] for c in CATEGORIES])[0]
    priority = rng.choices(PRIORITIES, weights=PRIORITY_GIVEN_CATEGORY[category])[0]

    vague = rng.random() < VAGUE_RATE
    ambiguous = (not vague) and rng.random() < AMBIGUOUS_RATE

    if vague:
        subject = rng.choice(VAGUE_SUBJECTS)
        sentences = [rng.choice(VAGUE_BODIES)]
        # one weak hint so the label is not pure chance
        if rng.random() < 0.6:
            sentences.append(_fill(rng.choice(CORE[category]), rng))
    else:
        subject = _capitalise(_fill(rng.choice(SUBJECTS[category]), rng))
        sentences = [_fill(s, rng) for s in rng.sample(CORE[category], k=rng.choice([1, 1, 2]))]

    if ambiguous:
        other = rng.choice([c for c in CATEGORIES if c != category])
        sentences.append(_fill(rng.choice(CORE[other]), rng))
        if rng.random() < 0.3:  # sometimes the subject line belongs to the other topic
            subject = _capitalise(_fill(rng.choice(SUBJECTS[other]), rng))

    if rng.random() < 0.5:
        sentences.append(_fill(rng.choice(DETAILS), rng))

    if rng.random() < PRIORITY_CUE_RATE:
        cue_level = _neighbour(priority, rng) if rng.random() < WRONG_CUE_RATE else priority
        sentences.append(rng.choice(PRIORITY_CUES[cue_level]))
        if cue_level == "urgent" and rng.random() < 0.4:
            subject = rng.choice(URGENT_SUBJECT_PREFIX) + subject

    rng.shuffle(sentences)
    sentences = [_capitalise(s) for s in sentences]
    opener, closer = rng.choice(OPENERS), rng.choice(CLOSERS)
    body = " ".join(part for part in [opener, *sentences, closer] if part)

    if rng.random() < TYPO_TICKET_RATE:
        body = _add_typos(body, rng)
        if rng.random() < 0.3:
            subject = _add_typos(subject, rng, rate=0.05)
    style = rng.random()
    if style < 0.12:
        subject, body = subject.lower(), body.lower()
    elif style < 0.15:
        subject = subject.upper()

    label_noise = 0
    if rng.random() < CATEGORY_LABEL_NOISE:
        category = rng.choice([c for c in CATEGORIES if c != category])
        label_noise = 1
    if rng.random() < PRIORITY_LABEL_NOISE:
        priority = _neighbour(priority, rng)
        label_noise = 1

    return {
        "ticket_id": f"T{ticket_id:05d}",
        "subject": subject,
        "body": body,
        "category": category,
        "priority": priority,
        "is_ambiguous": int(ambiguous),
        "is_vague": int(vague),
        "label_noise": label_noise,
    }


FIELDNAMES = ["ticket_id", "subject", "body", "category", "priority", "is_ambiguous", "is_vague", "label_noise"]


def generate(n: int = 3000, seed: int = 42) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    return [make_ticket(i + 1, rng) for i in range(n)]


def write_csv(rows: list[dict[str, str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" + explicit "\n" keeps the file byte-identical on Windows and Linux.
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/tickets.csv"))
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    write_csv(rows, args.out)
    noisy = sum(int(r["label_noise"]) for r in rows)
    ambiguous = sum(int(r["is_ambiguous"]) for r in rows)
    vague = sum(int(r["is_vague"]) for r in rows)
    print(f"wrote {len(rows)} tickets to {args.out} (seed={args.seed})")
    print(f"ambiguous={ambiguous} vague={vague} label_noise={noisy}")


if __name__ == "__main__":
    main()
