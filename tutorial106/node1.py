
def checkinput(state):
    request=state["messages"][0]
    print(request)
    if request == "yes":
        return {"messages":["yes"]}
    elif request == "no":
        return {"messages":["no"]}
    else:
        return{"messages":["NA"]}
