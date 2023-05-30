```mermaid
flowchart TD
	node1["chap01_analysis"]
	node2["chap01_download_data"]
	node3["chap01_sampling"]
	node2-->node1
	node2-->node3
	node3-->node1
	node4["chap02_ab_testing"]
```
