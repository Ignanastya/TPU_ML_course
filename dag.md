```mermaid
flowchart TD
	node1["catboost"]
	node2["data_preparation"]
	node3["decision_tree"]
	node4["linear_regression"]
	node5["neur_network"]
	node6["xgboost"]
	node2-->node1
	node2-->node3
	node2-->node4
	node2-->node5
	node2-->node6
	node4-->node1
	node4-->node3
	node4-->node6
```
