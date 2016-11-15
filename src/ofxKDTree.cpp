#include "ofxKDTree.h"


ofxKDTree::~ofxKDTree() {
    delete mat_index;
}

ofxKDTree::ofxKDTree() {
    mat_index = NULL;
}

void ofxKDTree::clear() {
    if (mat_index == NULL) {
        delete mat_index;
    }
    samples.clear();
}

void ofxKDTree::addPoint(vector<double> sample) {
    samples.push_back(sample);
}

void ofxKDTree::constructKDTree() {
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "Error: no samples");
        return;
    }
    dim = samples[0].size();
    for (auto s : samples) {
        if (s.size() != dim) {
            ofLog(OF_LOG_ERROR, "Error: samples don't all have the same size");
            return;
        }
    }
    mat_index = new KDTreeVectorOfVectorsAdaptor<vector<vector<double> >, double > (dim, samples, 10);
    mat_index->index->buildIndex();
}
    
void ofxKDTree::getKNN(vector<double> query_pt, int k, vector<size_t> & indexes, vector<double> & dists) {
    indexes.resize(k);
    dists.resize(k);
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(&indexes[0], &dists[0] );
    mat_index->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
}

void ofxKDTree::save(string filename) {
    const char * path = filename.c_str();
    FILE *f = fopen(path, "wb");
    if (!f) {
        throw std::runtime_error("Error writing index file!");
    }
    mat_index->index->saveIndex(f);
    fclose(f);
}

void ofxKDTree::load(string filename) {
    const char * path = filename.c_str();
    FILE *f = fopen(path, "rb");
    if (!f) {
        throw std::runtime_error("Error reading index file!");
    }
    mat_index = new KDTreeVectorOfVectorsAdaptor<vector<vector<double> >, double > (dim, samples, 10);
    mat_index->index->loadIndex(f);
    fclose(f);
}