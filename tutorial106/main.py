from langgraph.graph import Graph,StateGraph,END
from typing import TypedDict,Annotated,Sequence
import operator
from node1 import checkinput
from node2 import writeYes
from node3 import writeno

class AgentState(TypedDict):
    messages:Annotated[Sequence[str],operator.add]

workflow=StateGraph(AgentState)




workflow.add_node("node1",checkinput)
workflow.add_node("node2",writeYes)
workflow.add_node("node3",writeno)

def where_to_go(state):
    ctx=state["messages"]
    if ctx[1]=="yes":
        return "node2"
    elif ctx[1]=="no":
        return "node3"
    else:
        return "end"


workflow.set_entry_point("node1")
workflow.add_conditional_edges("node1",where_to_go,{
    "node2":"node2",
    "node3":"node3",
    "end":END
})
app=workflow.compile()

def getanswer(qsn):
    response=app.invoke({"messages":[qsn]})
    print(response["messages"][-1])


getanswer("no")