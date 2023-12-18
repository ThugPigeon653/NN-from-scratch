import sqlite3
import os
import glob
import plotly.graph_objects as go

class Display():
    nodes = []

    def __init__(self) -> None:
        pattern = "hidden-layers*.db"
        files = glob.glob(pattern)
        latest_db_file = max(files, key=os.path.getctime)
        self.conn = sqlite3.connect(latest_db_file)
        self.cursor = self.conn.cursor()

        for weight in self.get_all_weights():
            self.nodes.append((weight[2], weight[0]))
            self.nodes.append((weight[2] - 1, weight[1]))

        self.plot_network()

    def get_all_weights(self, batch_size=100) -> []:
        self.cursor.execute('SELECT * FROM weights')
        while True:
            batch = self.cursor.fetchmany(batch_size)
            if not batch:
                break
            yield batch

    def plot_network(self):
        fig = go.Figure()

        for node in self.nodes:
            fig.add_trace(go.Scatter(x=[node[0], node[2] - 1], y=[node[1], node[3]], mode='lines+markers'))

        fig.update_layout(title='Neural Network Structure', xaxis_title='Layer', yaxis_title='Node')

        fig.show()

# Create an instance of the Display class
Display()