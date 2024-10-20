# Environment-Adaptive Online Learning for Portable Energy Storage Based on Porous Electrode Model

The dynamic conditions and internal states of portable energy storage system (PESS), such as temperature, electricity price, state of charge (SOC), and state of health (SOH), significantly impact battery degradation. Current decision-making models for PESS operation often oversimplify the modeling of battery degradation. To address this, we introduce an environment-adaptive online learning framework that effectively integrates deep neural networks and reinforcement learning to exploit and explore external environments (i.e., electricity prices and temperature) and internal dynamics (i.e., battery degradation), providing decision support for PESS operation. This framework dynamically updates battery degradation and decision-making models in real-time, enhancing adaptive responses to external changes. Specifically, we developed a neural network based on porous electrode theory that considers multi-physical factors, such as charging power, initial and terminal SOC, SOH, and temperature to accurately assess battery degradation. This network is embedded within a deep reinforcement learning algorithm, enabling real-time, adaptive decision-making for PESS amidst varying environmental conditions. Furthermore, to navigate complex operational environments, a fine-tuning mechanism is incorporated into the degradation neural network. Application of this framework to the energy arbitrage of PESS in the California power grid demonstrates an average benefit increase of 37\% compared to traditional degradation assessment models.


# Team
This original version of the online learning algorithm was collaboratively developed by Yongkang Ding, Zhengrun Wu, and Xinjiang Chen.
