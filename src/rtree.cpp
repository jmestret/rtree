// [[Rcpp::depends(BH)]]

// Enable C++11 via this plugin to suppress 'long long' errors
// [[Rcpp::plugins("cpp11")]]

// Some of this code based on http://gallery.rcpp.org/articles/Rtree-examples/

#include <vector>
#include <Rcpp.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>
using namespace Rcpp;

// Mnemonics
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::point<double, 2, bg::cs::cartesian> point_t;
typedef bg::model::box<point_t> box;
typedef std::pair<point_t, unsigned int> value_t;

class RTreeCpp {
public:

  // Constructor, creates R-Tree on points in mat
  // mat must have 2 columns with x and y coordinates of points
  RTreeCpp(NumericMatrix mat) {
    coords = clone(mat);
    int size = mat.nrow();
    NumericVector x = mat(_, 0);
    NumericVector y = mat(_, 1);
    std::vector<value_t> point_value_pairs;

    // Create vector of point-index pairs
    for (int i = 0; i < size; ++i) {
      point_t point_t(x[i], y[i]);
      point_value_pairs.push_back(std::make_pair(point_t, static_cast<unsigned int>(i)));
    }

    // Create rtree using packing algorithm
    bgi::rtree<value_t, bgi::quadratic<16> > rtree_new(point_value_pairs);
    rtree_ = rtree_new;
  }

  // Within-box lookup
  // box_vec is [llx, lly, urx, ury] coordinates
  std::vector<int> intersects(NumericVector box_vec) {
    // Find points within box_vec
    box query_box(point_t(box_vec[0], box_vec[1]), point_t(box_vec[2], box_vec[3]));
    std::vector<value_t> result_n;
    rtree_.query(bgi::intersects(query_box), std::back_inserter(result_n));

    // Get just the indices of the points
    std::vector<int> indexes;
    std::vector<value_t>::iterator itr;
    for (itr = result_n.begin(); itr != result_n.end(); ++itr) {
      value_t value = *itr;
      indexes.push_back(value.second);
    }
    return indexes;
  }

  // Get Haversine distances between point and points referenced by their indexes
  std::vector<double> get_haversine_distances(NumericVector point_vec, std::vector<int> indexes, double earth_radius) {
    double lat1 = point_vec[1] * M_PI / 180.0; // Latitude
    double lon1 = point_vec[0] * M_PI / 180.0; // Longitude

    std::vector<double> distances;

    for (unsigned int i = 0; i < indexes.size(); i++) {
      double lat2 = coords(indexes[i], 1) * M_PI / 180.0; // Latitude
      double lon2 = coords(indexes[i], 0) * M_PI / 180.0; // Longitude

      // Haversine formula
      double dlat = lat2 - lat1;
      double dlon = lon2 - lon1;

      double a = std::pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * std::pow(sin(dlon / 2), 2);
      double c = 2 * atan2(sqrt(a), sqrt(1 - a));
      double distance = earth_radius * c; // Distance in kilometers

      distances.push_back(distance);
    }

    return distances;
  }

  // Get Euclidean distances between point and points referenced by their indexes
  std::vector<double> get_euclidean_distances(NumericVector point_vec, std::vector<int> indexes) {
    std::vector<double> distances;

    for (unsigned int i = 0; i < indexes.size(); i++) {
      double lat2 = coords(indexes[i], 1); // Latitude
      double lon2 = coords(indexes[i], 0); // Longitude

      double distance = sqrt(pow(lon2 - point_vec[0], 2) + pow(lat2 - point_vec[1], 2)); // Distance in degrees
      distances.push_back(distance);
    }

    return distances;
  }

  // Get indices of points within distance of point
// Note: Returns R indices (starting at 1, not 0!)
std::vector<int> within_distance(NumericVector point_vec, double distance, std::string distance_metric, double earth_radius) {
    // Candidate points are within a square with side 2 * distance
    std::vector<int> indist_indexes;
    NumericVector box_vec = NumericVector::create(
      point_vec[0] - distance, point_vec[1] - distance,
      point_vec[0] + distance, point_vec[1] + distance);

    std::vector<int> inbox_indexes = intersects(box_vec);
    if (inbox_indexes.size() == 0) {
      return indist_indexes;
    }

    // Filter candidates by the actual distance
    std::vector<double> inbox_distances;
    if (distance_metric == "haversine") {
      inbox_distances = get_haversine_distances(point_vec, inbox_indexes, earth_radius);
    } else if (distance_metric == "euclidean") {
      inbox_distances = get_euclidean_distances(point_vec, inbox_indexes);
    } else {
      Rcpp::stop("Invalid distance metric. Use 'haversine' or 'euclidean'.");
    }

    for (unsigned int i = 0; i < inbox_distances.size(); i++) {
      if (inbox_distances[i] <= distance) {
        indist_indexes.push_back(inbox_indexes[i] + 1);
      }
    }
    return indist_indexes;
}

// Multi point version of within_distance
// Note: Returns R indices (starting at 1, not 0!)
List within_distance_list(NumericMatrix point_mat, double distance, std::string distance_metric, double earth_radius) {
    List indist_indexes_ls(point_mat.nrow());
    for (int i = 0; i < point_mat.nrow(); i++) {
      NumericVector point_vec = point_mat(i, _);
      std::vector<int> indist_indexes = within_distance(point_vec, distance, distance_metric, earth_radius);
      indist_indexes_ls(i) = wrap(indist_indexes);
    }
    return indist_indexes_ls;
}


  // Counting version of within_distance_list
  // For each point in point_mat, returns the number of points within distance.
  // This uses much less memory than within_distance_list when point_mat is
  // large and there are many points within distance.
  std::vector<int> count_within_distance_list(NumericMatrix point_mat, double distance, std::string distance_metric, double earth_radius) {
    std::vector<int> indist_indexes_ls(point_mat.nrow());
    for (int i = 0; i < point_mat.nrow(); i++) {
      NumericVector point_vec = point_mat(i, _);
      std::vector<int> indist_indexes = within_distance(point_vec, distance, distance_metric, earth_radius);
      indist_indexes_ls[i] = indist_indexes.size();
    }
    return indist_indexes_ls;
  }

  // KNN
  // Note: Returns R indices (starting at 1, not 0!)
  std::vector<int> knn(NumericVector point, unsigned int n) {
    std::vector<value_t> result_n;
    rtree_.query(bgi::nearest(point_t(point[0], point[1]), n), std::back_inserter(result_n));
    std::vector<int> indexes;
    std::vector<value_t>::iterator itr;
    for (itr = result_n.begin(); itr != result_n.end(); ++itr) {
      value_t value = *itr;
      indexes.push_back(value.second + 1);
    }
    return indexes;
  }

  // Multi point version of KNN
  // Note: Returns R indices (starting at 1, not 0!)
  List knn_list(NumericMatrix point_mat, unsigned int n) {
    List knn_indexes_ls(point_mat.nrow());
    for (int i = 0; i < point_mat.nrow(); i++) {
      NumericVector point_vec = point_mat(i, _);
      std::vector<int> knn_indexes = knn(point_vec, n);
      knn_indexes_ls(i) = wrap(knn_indexes);
    }
    return knn_indexes_ls;
  }

private:
  bgi::rtree<value_t, bgi::quadratic<16>> rtree_;
  NumericMatrix coords;
};

RCPP_MODULE(rtreecpp) {
  class_<RTreeCpp>("RTreeCpp")

  .constructor<NumericMatrix>()

  .method("intersects", &RTreeCpp::intersects)
  .method("within_distance", &RTreeCpp::within_distance)
  .method("within_distance_list", &RTreeCpp::within_distance_list)
  .method("count_within_distance_list", &RTreeCpp::count_within_distance_list)
  .method("knn", &RTreeCpp::knn)
  .method("knn_list", &RTreeCpp::knn_list);
}

