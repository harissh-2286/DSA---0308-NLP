grammar={
    "S":["NP VP"],
    "NP":["Det N"],
    "VP":["V"],
    "Det":["the"],
    "N_singular":["boy","girl","student"],
    "N_plural":["boys","girls","students"],
    "V_singular":["runs","plays","eats"],
    "V_plural":["run","play","eat"]
    }
def check_agreement(sentence):
    words=sentence.lower().split()
    if len(words)!=3:
        return "Reject: Has less words"
    determinent,noun,verb=words
    if noun in grammar["N_singular"] and verb in grammar["V_singular"]:
        return "Accepted, Verb and Noun are singular and agree with each other"
    if noun in grammar["N_plural"] and verb in grammar["V_plural"]:
        return "Accepted, Verb and Noun are plural and agree with each other"
    return "Rejected, Verb and Noun don't agree with each other"
sentences=[
    "The boy runs",
    "The boys run",
    "The girl runs",
    "The girls run",
    "The students play",
    "The student plays",
    "The boy plays",
    "the boys play",
    "The girl plays",
    "The girls play",
    "the student runs",
    "the students run",
    ]
for sentence in sentences:
    print("Check Agreement for: ",sentence," ->",check_agreement(sentence),"\n")
