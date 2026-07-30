states = {"q0", "q1", "q2"}
alphabet = {"a", "b"}
transition = {
    "q0": {"a": "q1", "b": "q0"},
    "q1": {"a": "q1", "b": "q2"},
    "q2": {"a": "q1", "b": "q0"}
}
initial_state = "q0"
final_states = {"q2"}
n = int(input("Enter number of strings to test: "))
for i in range(n):
    string = input(f"\nEnter String {i+1}: ")
    current_state = initial_state
    path = [current_state]
    valid = True
    for symbol in string:
        if symbol not in alphabet:
            print("Invalid Input Symbol:", symbol)
            valid = False
            break
        current_state = transition[current_state][symbol]
        path.append(current_state)
    if valid:
        print("Transition Path:")
        print(" → ".join(path))

        if current_state in final_states:
            print("Accepted")
        else:
            print("Rejected")
