from diagrams import Cluster, Diagram
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Client

with Diagram("ScanMate Architecture", show=False):
    api_client = Client("API Client")

    with Cluster("Kong Gateway"):
        router = Server("Router")
        rate_limiter = Server("Rate Limiter")
        load_balancer = Server("Load Balancer")

    with Cluster("Services"):
        with Cluster("ScanMate Engine"):
            server_one = Server("server-one")
            server_two = Server("server-two")
            server_three = Server("server-three")
        with Cluster("Github Pages"):
            docs = Server("docs")

    api_client >> rate_limiter
    rate_limiter >> router
    router >> load_balancer
    load_balancer >> server_one
    load_balancer >> server_two
    load_balancer >> server_three
    router >> docs