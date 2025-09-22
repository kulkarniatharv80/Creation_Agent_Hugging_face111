# graphs/email_workflow.py

from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END

# Step 1: Define your state (workflow data)
@dataclass
class EmailState:
    subject: str = ""
    body: str = ""
    recipient: str = ""

# Step 2: Define functions (nodes)
def draft_email(state: EmailState) -> EmailState:
    state.subject = "Meeting Reminder"
    state.body = "This is a reminder for our meeting tomorrow at 10 AM."
    return state

def review_email(state: EmailState) -> EmailState:
    state.body += "\n\nReviewed and approved."
    return state

def send_email(state: EmailState) -> EmailState:
    print(f"Sending email to {state.recipient}:\nSubject: {state.subject}\n{state.body}")
    return state

# Step 3: Build the workflow graph
builder = StateGraph(EmailState)

builder.add_node("draft", draft_email)
builder.add_node("review", review_email)
builder.add_node("send", send_email)

builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", "send")
builder.add_edge("send", END)

# Step 4: Compile the graph
graph = builder.compile()
