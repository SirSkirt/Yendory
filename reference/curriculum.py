"""
Synthetic teaching curriculum for Nova-2, in Nova's own dialogue format:
    "U: <question>\nN: <answer>\n"

Lessons: greetings/identity, colour facts, "who has more/fewer", "where is it", yes/no.
Names are randomly generated and kept OUT of the word vocabulary, so Nova reads and writes
them character by character. TEST names come from a separate generator and never appear in
training, so any correct answer about them has to come from reading the prompt, not memory.
Everything here is training data / evaluation keys; nothing runs during Nova's inference.
"""
import random

COLORS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "brown", "pink"]
OBJECTS = ["ball", "cup", "hat", "book", "bag", "key", "box", "lamp", "pen", "flag", "bell", "plate", "stone"]
PLURALS = ["apples", "books", "cups", "keys", "balls", "bags", "stones", "plates", "pens", "coins", "shells", "cards"]
PLACES = ["box", "bag", "jar", "drawer", "basket", "cup", "car", "house", "garden", "kitchen"]
THINGS = ["key", "ball", "book", "coin", "map", "pen", "ring", "toy", "card", "shell"]

GREETINGS = [  # (question variants, answer)
    (["hello", "Hello", "hello nova", "Hello Nova", "hello!", "Hello!"], "Hello!"),
    (["hi", "Hi", "hi nova", "Hi Nova", "hi!", "hey"], "Hi!"),
    (["good morning", "Good morning", "good morning nova", "Good morning!"], "Good morning!"),
    (["good evening", "Good evening", "good evening nova"], "Good evening!"),
    (["good night", "Good night", "good night nova"], "Good night!"),
    (["how are you", "How are you?", "how are you nova", "How are you doing?", "how are you today"], "I am well, thank you."),
    (["what is your name", "What is your name?", "what's your name", "What is your name"], "My name is Nova."),
    (["who are you", "Who are you?", "who are you?", "what are you"], "I am Nova."),
    (["thank you", "Thank you", "thanks", "Thanks!", "thank you nova"], "You're welcome."),
    (["goodbye", "Goodbye", "bye", "Bye!", "see you later"], "Goodbye!"),
    (["nice to meet you", "Nice to meet you", "nice to meet you nova"], "Nice to meet you too."),
]
# Paraphrases held out from training (greeting generalization test)
GREETING_TEST = [("Hello there", "Hello!"), ("hi there", "Hi!"), ("What is your name nova?", "My name is Nova."),
                 ("thanks nova", "You're welcome."), ("goodbye nova", "Goodbye!"), ("how are you doing today", "I am well, thank you.")]

ON, VO = "bdfgklmnprstvzj", "aeiou"


def make_names(seed, n, exclude=()):
    rng = random.Random(seed); out, seen = [], set(exclude)
    while len(out) < n:
        s = "".join(rng.choice(ON) + rng.choice(VO) for _ in range(rng.choice([2, 2, 3])))
        if rng.random() < 0.4:
            s += rng.choice("nlrs")
        s = s.capitalize()
        if s not in seen:
            seen.add(s); out.append(s)
    return out


TRAIN_NAMES = make_names(1, 600)
TEST_NAMES = make_names(999, 80, exclude=TRAIN_NAMES)   # never used in training


def vocabulary():
    """Common words only (3-24 letters, as tokenizer.rs requires). Names are deliberately absent."""
    words = set()
    for w in COLORS + OBJECTS + PLURALS + PLACES + THINGS:
        words.add(w); words.add(w.capitalize())
    words |= {"What", "color", "Who", "has", "have", "more", "fewer", "Where", "the", "The", "Does",
              "Yes", "nova", "Nova", "hello", "Hello", "good", "Good", "morning", "evening", "night",
              "how", "How", "are", "you", "your", "name", "thank", "Thank", "thanks", "welcome",
              "goodbye", "Goodbye", "nice", "meet", "too", "well", "who", "what", "My", "You're"}
    return sorted(w for w in words if 3 <= len(w) <= 24 and w.replace("'", "").isalnum())


def lesson(rng, names, kind=None):
    kind = kind or rng.choices(["greet", "color", "compare", "where", "yesno"], [2, 3, 4, 3, 2])[0]
    if kind == "greet":
        qs, a = rng.choice(GREETINGS); return kind, rng.choice(qs), a
    A, B = rng.sample(names, 2)
    if kind == "color":
        c1, c2 = rng.sample(COLORS, 2); o1, o2 = rng.sample(OBJECTS, 2)
        if rng.random() < 0.35:
            return kind, f"{A} has a {c1} {o1}. What color is {A}'s {o1}?", f"{c1.capitalize()}."
        facts = [f"{A} has a {c1} {o1}.", f"{B} has a {c2} {o2}."]; rng.shuffle(facts)
        who, obj, col = rng.choice([(A, o1, c1), (B, o2, c2)])
        return kind, f"{facts[0]} {facts[1]} What color is {who}'s {obj}?", f"{col.capitalize()}."
    if kind == "compare":
        x, y = rng.sample(range(1, 10), 2); obj = rng.choice(PLURALS)
        if rng.random() < 0.7:
            return kind, f"{A} has {x} {obj}. {B} has {y} {obj}. Who has more {obj}?", f"{A if x > y else B}."
        return kind, f"{A} has {x} {obj}. {B} has {y} {obj}. Who has fewer {obj}?", f"{A if x < y else B}."
    if kind == "where":
        t1, t2 = rng.sample(THINGS, 2); p1, p2 = rng.sample(PLACES, 2)
        if rng.random() < 0.35:
            return kind, f"The {t1} is in the {p1}. Where is the {t1}?", f"In the {p1}."
        facts = [f"The {t1} is in the {p1}.", f"The {t2} is in the {p2}."]; rng.shuffle(facts)
        t, p = rng.choice([(t1, p1), (t2, p2)])
        return kind, f"{facts[0]} {facts[1]} Where is the {t}?", f"In the {p}."
    # yes/no
    c, o = rng.choice(COLORS), rng.choice(OBJECTS)
    if rng.random() < 0.5:
        return kind, f"{A} has a {c} {o}. Does {A} have a {o}?", "Yes."
    o2 = rng.choice([x for x in OBJECTS if x != o])
    return kind, f"{A} has a {c} {o}. Does {A} have a {o2}?", "No."


def fmt(q, a=None):
    return f"U: {q}\nN: " + ("" if a is None else a + "\n")
