def writeno(state):
    print(state)
    input=state["messages"][-1]
    return {"messages":["user entered no"]}