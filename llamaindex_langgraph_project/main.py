# main.py

from graphs.email_workflow import graph, EmailState

# Run the graph
state = EmailState(recipient="kulkarniatharv929@gmail.com")

result = graph.invoke(state)

print("Workflow completed ✅")
