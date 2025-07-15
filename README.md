

# Networked-Vaccination-Game-With-The-Disturbance-Of-Infectious-Sources
The social dilemma of voluntary vaccination as a public goods dilemma has obtained extensive attention from researchers. Prior networked vaccination game studies often assume infectious sources are randomly distributed. However, the outbreak of an epidemic is often associated with various factors such as spatial location, individual heterogeneity, and human interference. To clarify the crucial role of infectious sources in the population of vaccination behaviors and epidemic scale, we propose a networked vaccination game model incorporating disturbances in infectious source. Several infectious source selection strategies are designed from the methods of node centrality, key node evaluation, and influence maximization. Results show that for moderate vaccination cost, these strategies lead to significantly different vaccination behaviors. Focusing on two representative strategies, we identify the contagion ability of the infectious source selection strategy as a key factor influencing the result of networked vaccination game. Specifically, the strategy with stronger propagation capability effectively promotes vaccination, whereas weaker ones suppress it. State transition analysis reveals distinct mechanisms behind these strategies. Furthermore, treating infectious source disturbance as a short-term local shock, we assess the resilience of networked vaccination game via robustness and adaptability from a social system perspective. Experimental results indicate poor resilience when facing disturbances from the strategy with strong propagation capability or pandemic fatigue. Conversely, when encountering the infectious source disturbance with weaker propagation capacity, the system demonstrates good resilience, and the result of the disturbed system can nearly converge to the original system. We hope these findings can shed light on how social systems recover from the epidemic risk with the shock of infectious sources.

This repository hosts the source code of **Impact of infectious sources on the vaccination dilemma in networked population**, which will be published in **Chaos, Solitons & Fractals**.

## Requirements
It is worth mentioning that because python runs slowly, we use **numba** library to improve the speed of code running.
* networkx==3.1
* numba==0.57.0
* numpy==1.23.0
* pandas==2.0.2
* scipy==1.11.1
* seaborn==0.12.2
* tqdm==4.65.0

## Setup
The installation of Networked-Vaccination-Game-With-The-Disturbance-Of-Infectious-Sources is very easy. We've tested our code on Python 3.10 and above. We strongly recommend using conda to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.10 based conda environment.
****
```
conda create -n networked_vac_game python==3.10.2 -y
conda activate networked_vac_game
pip install -r requirements.txt
```

## Running
```
python run.py
```


## Contact
Please email [Jingrui Wang](https://scholar.google.com/citations?user=oiu-yTYAAAAJ&hl=en)(wangjingrui530@gmail.com) for further questions.
