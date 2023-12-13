CREATE TABLE layer (
    layerId INTEGER PRIMARY KEY,
    activation_type VARCHAR
);

CREATE TABLE node (
    layerId INTEGER,
    nodeId INTEGER,
    PRIMARY KEY (layerId, nodeId),
    FOREIGN KEY (layerId) REFERENCES layer(layerId)
);

CREATE TABLE weights (
    fromNodeId INTEGER,
    toNodeId INTEGER,
    layerId INTEGER,
    weight NUMBER,
    PRIMARY KEY (fromNodeId, toNodeId, layerId),
    FOREIGN KEY (fromNodeId) REFERENCES node(nodeId),
    FOREIGN KEY (toNodeId) REFERENCES node(nodeId),
    FOREIGN KEY (layerId) REFERENCES node(layerId)
);