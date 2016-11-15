#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    int nSamples = 10000;
    int dim = 10;
    
    // Generate points:
    for (int i=0; i<nSamples; i++) {
        vector<double> sample(dim);
        for (int d=0;d<dim;d++) {
            sample[d] = ofRandom(1);
        }
        samples.push_back(sample);
        kdTree.addPoint(sample);
    }
    
    // construct the KD Tree
    kdTree.constructKDTree();
    
    // save the index
    kdTree.save(ofToDataPath("index.bin"));
    
    // or instead of constructing it, can load the index from saved file
    //kdTree.load(ofToDataPath("index.bin"));
    
    
    // make a random query point
    vector<double> query_pt(dim);
    for (int d=0; d<dim; d++) {
        query_pt[d] = ofRandom(1);
    }
    cout << "query point: "<<ofToString(query_pt) <<endl;
    
    // get 3 nearest-neighbors
    vector<size_t> indexes;
    vector<double> dists;
    kdTree.getKNN(query_pt, 3, indexes, dists);
    
    for (int i=0; i<3; i++) {
        cout << "nearest point: "<<ofToString(samples[indexes[i]]) << " : distance " << dists[i] << endl;
    }
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
