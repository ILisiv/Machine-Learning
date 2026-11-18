# Individual Project Proposal: Lightweight Sentiment Analysis for Mobile App Reviews

* **Student Name:** Volkov Illia
* **Student ID:** 76176
* **Problem:** Deploying natural language processing models on smartphone devices to classify customer review sentiment locally without sending sensitive user data to cloud servers.
* **Dataset:** SST-2 (Stanford Sentiment Treebank) or Amazon App Reviews subset.
* **Success Metric:** Validation accuracy $\geq 85\%$ with single-thread CPU inference latency $\leq 50\text{ ms}$ and model size $\leq 5\text{ MB}$.
* **Deployment Constraints:** Edge memory footprint $\leq 5\text{ MB}$, single-thread CPU inference latency under $50\text{ ms}$.
* **Planned Optimization:** Post-training quantization (INT8) and embedding compression to shrink memory size and speed up CPU inference.