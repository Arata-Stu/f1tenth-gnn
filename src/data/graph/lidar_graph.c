#include <Python.h>
#include <stdlib.h>
#include <math.h>

#define DEG2RAD (M_PI / 180.0)
#define MAX_EDGES_PER_NODE 5

typedef struct {
    double x;
    double y;
} Node;

typedef struct {
    int from;
    int to;
    double weight;
} Edge;

static Node** nodes = NULL;
static Edge** edges = NULL;
static int* edge_counts = NULL;
static int num_nodes = 0;
static int batch_size = 0;
static int k_neighbors = 3;
static double edge_threshold = 1.0;
static int use_knn = 1; // 1: use KNN, 0: use full search

static PyObject* initialize(PyObject* self, PyObject* args) {
    int new_num_nodes, new_batch_size, new_k_neighbors;
    double new_edge_threshold;
    int new_use_knn;
    if (!PyArg_ParseTuple(args, "iiidi", &new_num_nodes, &new_batch_size, &new_k_neighbors, &new_edge_threshold, &new_use_knn)) {
        return NULL;
    }

    if (nodes) {
        for (int i = 0; i < batch_size; i++) {
            free(nodes[i]);
            free(edges[i]);
        }
        free(nodes);
        free(edges);
        free(edge_counts);
    }

    num_nodes = new_num_nodes;
    batch_size = new_batch_size;
    k_neighbors = new_k_neighbors;
    edge_threshold = new_edge_threshold;
    use_knn = new_use_knn;

    nodes = (Node**)malloc(batch_size * sizeof(Node*));
    edges = (Edge**)malloc(batch_size * sizeof(Edge*));
    edge_counts = (int*)malloc(batch_size * sizeof(int));

    for (int b = 0; b < batch_size; b++) {
        nodes[b] = (Node*)malloc(num_nodes * sizeof(Node));
        edges[b] = (Edge*)malloc(num_nodes * MAX_EDGES_PER_NODE * sizeof(Edge));
        edge_counts[b] = 0;
    }

    Py_RETURN_NONE;
}

static PyObject* build_graph(PyObject* self, PyObject* args) {
    PyObject* input_lists;
    if (!PyArg_ParseTuple(args, "O", &input_lists)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists as input");
        return NULL;
    }
    if (!PyList_Check(input_lists)) {
        PyErr_SetString(PyExc_TypeError, "Input is not a list of lists");
        return NULL;
    }
    if (PyList_Size(input_lists) != batch_size) {
        PyErr_SetString(PyExc_ValueError, "Batch size mismatch");
        return NULL;
    }

    double angle_increment = 270.0 / num_nodes;

    for (int b = 0; b < batch_size; b++) {
        PyObject* scan = PyList_GetItem(input_lists, b);
        if (!PyList_Check(scan)) {
            PyErr_SetString(PyExc_TypeError, "Sub-item is not a list");
            return NULL;
        }
        edge_counts[b] = 0;

        for (int i = 0; i < num_nodes; i++) {
            double dist = PyFloat_AsDouble(PyList_GetItem(scan, i));
            double angle = (-135.0 + i * angle_increment) * DEG2RAD;
            nodes[b][i].x = dist * cos(angle);
            nodes[b][i].y = dist * sin(angle);
        }

        for (int i = 0; i < num_nodes; i++) {
            int count = 0;
            double* distances = (double*)malloc(num_nodes * sizeof(double));
            int* indices = (int*)malloc(num_nodes * sizeof(int));

            for (int j = 0; j < num_nodes; j++) {
                if (i == j) continue;
                double dx = nodes[b][j].x - nodes[b][i].x;
                double dy = nodes[b][j].y - nodes[b][i].y;
                double d = sqrt(dx * dx + dy * dy);
                if (d < edge_threshold) {
                    distances[count] = d;
                    indices[count] = j;
                    count++;
                }
            }

            for (int m = 0; m < count - 1; m++) {
                for (int n = m + 1; n < count; n++) {
                    if (distances[n] < distances[m]) {
                        double temp_d = distances[m];
                        int temp_i = indices[m];
                        distances[m] = distances[n];
                        indices[m] = indices[n];
                        distances[n] = temp_d;
                        indices[n] = temp_i;
                    }
                }
            }

            int k_limit = use_knn ? k_neighbors : count;
            for (int k = 0; k < k_limit && k < count && edge_counts[b] < num_nodes * MAX_EDGES_PER_NODE; k++) {
                edges[b][edge_counts[b]].from = i;
                edges[b][edge_counts[b]].to = indices[k];
                edges[b][edge_counts[b]].weight = distances[k];
                edge_counts[b]++;
            }

            free(distances);
            free(indices);
        }
    }

    PyObject* batch_edge_list = PyList_New(batch_size);
    for (int b = 0; b < batch_size; b++) {
        PyObject* edge_list = PyList_New(edge_counts[b]);
        for (int i = 0; i < edge_counts[b]; i++) {
            PyObject* tpl = PyTuple_New(3);
            PyTuple_SetItem(tpl, 0, PyLong_FromLong(edges[b][i].from));
            PyTuple_SetItem(tpl, 1, PyLong_FromLong(edges[b][i].to));
            PyTuple_SetItem(tpl, 2, PyFloat_FromDouble(edges[b][i].weight));
            PyList_SetItem(edge_list, i, tpl);
        }
        PyList_SetItem(batch_edge_list, b, edge_list);
    }
    return batch_edge_list;
}

static PyObject* get_node_positions(PyObject* self, PyObject* args) {
    PyObject* batch_node_list = PyList_New(batch_size);
    for (int b = 0; b < batch_size; b++) {
        PyObject* node_list = PyList_New(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            PyObject* coords = PyTuple_New(2);
            PyTuple_SetItem(coords, 0, PyFloat_FromDouble(nodes[b][i].x));
            PyTuple_SetItem(coords, 1, PyFloat_FromDouble(nodes[b][i].y));
            PyList_SetItem(node_list, i, coords);
        }
        PyList_SetItem(batch_node_list, b, node_list);
    }
    return batch_node_list;
}

static PyMethodDef LidarGraphMethods[] = {
    {"initialize", initialize, METH_VARARGS, "Initialize graph size, batch size, k-neighbors, threshold, and mode (KNN or full)."},
    {"build_graph", build_graph, METH_VARARGS, "Build graph from LiDAR data."},
    {"get_node_positions", get_node_positions, METH_NOARGS, "Get node positions per batch."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lidargraphmodule = {
    PyModuleDef_HEAD_INIT,
    "lidar_graph",
    NULL,
    -1,
    LidarGraphMethods
};

PyMODINIT_FUNC PyInit_lidar_graph(void) {
    return PyModule_Create(&lidargraphmodule);
}