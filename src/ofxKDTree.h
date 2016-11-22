#pragma once

#include "ofMain.h"
#include <nanoflann.hpp>

using namespace nanoflann;


template <class VectorOfVectorsType, typename num_t = double, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType,num_t,DIM,Distance> self_t;
    typedef typename Distance::template traits<num_t,self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t,self_t,DIM,IndexType>  index_t;
    
    index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.
    
    /// Constructor: takes a const ref to the vector of vectors object with the data points
    KDTreeVectorOfVectorsAdaptor(const int dimensionality, const VectorOfVectorsType &mat, const int leaf_max_size = 10) : m_data(mat)
    {
        cout << mat.size() << " " << mat[0].size() << endl;
        assert(mat.size()!=0 && mat[0].size()!=0);
        const size_t dims = mat[0].size();
        if (DIM>0 && static_cast<int>(dims)!=DIM)
            throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
        index = new index_t( dims, *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size ) );
        index->buildIndex();
    }
    
    ~KDTreeVectorOfVectorsAdaptor() {
        delete index;
    }
    
    const VectorOfVectorsType &m_data;
    
    /** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.
     * \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
     */
    inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
    {
        nanoflann::KNNResultSet<num_t,IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }
    
    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */
    
    const self_t & derived() const {
        return *this;
    }
    self_t & derived()       {
        return *this;
    }
    
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return m_data.size();
    }
    
    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline num_t kdtree_distance(const num_t *p1, const size_t idx_p2,size_t size) const
    {
        num_t s=0;
        for (size_t i=0; i<size; i++) {
            const num_t d= p1[i]-m_data[idx_p2][i];
            s+=d*d;
        }
        return s;
    }
    
    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, int dim) const {
        return m_data[idx][dim];
    }
    
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const {
        return false;
    }
    
    /** @} */
    
}; // end of KDTreeVectorOfVectorsAdaptor



class ofxKDTree
{
public:
    ofxKDTree();
    ~ofxKDTree();
    
    void clear();
    
    void addPoint(vector<double> sample);
    void constructKDTree();
    
    void getKNN(vector<double> query_pt, int k, vector<size_t> & indexes, vector<double> & dists);
    
    void save(string filename);
    void load(string filename);
    
    vector<vector<double> > & getSamples(){return samples;}

private:
    bool checkSamples();
    
    KDTreeVectorOfVectorsAdaptor<vector<vector<double> >, double > *mat_index;
    vector<vector<double> > samples;
    int dim;
};

