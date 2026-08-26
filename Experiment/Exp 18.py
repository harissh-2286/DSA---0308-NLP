import re

def parse_fopc(expression):
    print("\nOriginal Expression:")
    print(expression)

    # Find logical operators
    operators = re.findall(r'\b(AND|OR|NOT|IMPLIES)\b', expression.upper())

    # Find predicates and their arguments
    predicates = re.findall(r'([A-Za-z]+)\s*\(([^()]*)\)', expression)

    # Find variables and constants
    arguments = []

    for predicate, args in predicates:
        args_list = [arg.strip() for arg in args.split(",")]

        for arg in args_list:
            arguments.append(arg)

    print("\nParsed Result")
    print("-" * 40)

    print("Predicates:")
    for predicate, args in predicates:
        print("  ", predicate, "->", args.split(","))

    print("\nArguments:")
    for arg in arguments:
        arg = arg.strip()

        # Variables start with lowercase letters
        if arg.islower():
            print("  ", arg, "-> Variable")
        else:
            print("  ", arg, "-> Constant")

    print("\nLogical Operators:")
    if operators:
        for op in operators:
            print("  ", op)
    else:
        print("   No logical operators found")

    print("\nFOPC expression parsed successfully.")


# Get input from user
expression = input("Enter a FOPC expression: ")

parse_fopc(expression)
