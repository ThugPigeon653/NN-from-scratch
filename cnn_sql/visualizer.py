import sqlite3
import os
import glob
import matplotlib.pyplot as plt

class Display():
    nodes = []

    def __init__(self) -> None:
        pattern = "hidden-layers*.db"
        files = glob.glob(pattern)
        latest_db_file = max(files, key=os.path.getctime)
        self.conn = sqlite3.connect(latest_db_file)
        self.cursor = self.conn.cursor()

        for weight in self.get_all_nodes():
            self.nodes.append((weight[2], weight[0]))
            self.nodes.append((weight[2] - 1, weight[1]))

        self.plot_network()

    def get_all_nodes(self, batch_size=100) -> []:
        self.cursor.execute('SELECT * FROM node')
        while True:
            batch = self.cursor.fetchmany(batch_size)
            if not batch:
                break
            yield batch

    def plot_network(self):
        plt.figure(figsize=(10, 5))
        for node in self.nodes:
            plt.plot(node, marker='o', color='b')

        plt.title('Neural Network Structure')
        plt.xlabel('Layer')
        plt.ylabel('Node')
        plt.grid(True)

        plt.imsave('plot.png')

# Create an instance of the Display class
Display()
