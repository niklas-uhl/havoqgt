#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <havoqgt/delegate_partitioned_graph.hpp>
#include <havoqgt/distributed_db.hpp>
#include <havoqgt/parallel_edge_list_reader.hpp>
#include <filesystem>
#include <kagen.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <havoqgt/cache_utilities.hpp>
namespace havoqgt::extension {

enum class Generator {
  none,
  rgg_2d,
  rdg_2d,
  gnm,
  rmat,
  rhg
};

struct Config {
  Generator gen = Generator::none;
  unsigned long long gen_n = 10;
  unsigned long long gen_m = 14;
  double gen_gamma = 3.0;
  bool read_directly = false;
  bool allocate_directly = false;
  uint64_t delegate_threshold = 1048576;
  double gbyte_per_rank   = 0.25;
  uint64_t partition_passes = 1;
  uint64_t chunk_size       = 8 * 1024;
  bool undirected       = false;
};

const std::map<std::string, Generator> gen_map = {
    {"none", Generator::none},     {"rgg_2d", Generator::rgg_2d},
    {"rdg_2d", Generator::rdg_2d}, {"gnm", Generator::gnm},
    {"rmat", Generator::rmat},     {"rhg", Generator::rhg}};

struct BinaryGraphEdgeAdaptor {
  using value_type = std::tuple<uint64_t, uint64_t, double>;
  BinaryGraphEdgeAdaptor(std::string const& file, MPI_Comm comm) {
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    auto base = std::filesystem::path(file);
    auto path = base.parent_path();
    auto basename = base.stem();
    auto first_out_path = path / (basename.string() + ".first_out");
    auto first_out_filesize = std::filesystem::file_size(first_out_path);
    first_out_size = first_out_filesize / sizeof(uint64_t);
    auto head_path = path / (basename.string() + ".head");
    auto head_filesize = std::filesystem::file_size(head_path);
    head_size = head_filesize / sizeof(uint64_t);
    first_out_fd = open(first_out_path.c_str(), O_RDONLY);
    void* ptr     = mmap(NULL, first_out_filesize, PROT_READ, MAP_PRIVATE, first_out_fd, 0);
    if (ptr == MAP_FAILED) {
      fprintf(stderr, "%s\n", strerror(errno));
      std::exit(1);
    }
    first_out = reinterpret_cast<uint64_t*>(ptr);
    head_fd = open(head_path.c_str(), O_RDONLY);
    ptr = mmap(NULL, head_filesize, PROT_READ, MAP_PRIVATE, head_fd, 0);
    if (ptr == MAP_FAILED) {
      fprintf(stderr, "%s\n", strerror(errno));
      std::exit(1);
    }
    head = reinterpret_cast<uint64_t*>(ptr);

  }
  BinaryGraphEdgeAdaptor(BinaryGraphEdgeAdaptor const&) = delete;
  BinaryGraphEdgeAdaptor(BinaryGraphEdgeAdaptor&&) = delete;
  ~BinaryGraphEdgeAdaptor() {
    munmap(first_out, first_out_size * sizeof(uint64_t));
    munmap(head, head_size * sizeof(uint64_t));
    close(first_out_fd);
    close(head_fd);
  }
  struct EdgeIterator {
    EdgeIterator(uint64_t node_id, uint64_t edge_id, uint64_t* first_out,
                 uint64_t* head)
        : node_id(node_id),
          edge_id(edge_id),
          first_out(first_out),
          head(head) {}
    std::tuple<uint64_t, uint64_t, double> operator*() {
      return std::make_tuple(node_id, head[edge_id], 0.0);
    }
    bool operator!=(EdgeIterator const& other) const {
      return this->edge_id != other.edge_id;
    }
    EdgeIterator operator++(int) {
        auto previous = *this;
        this->edge_id++;
        if (this->edge_id >= first_out[node_id + 1]) {
          this->node_id++;
        }
        return previous;
    }
    EdgeIterator& operator++() {
        this->edge_id++;
        if (this->edge_id >= first_out[node_id + 1]) {
          this->node_id++;
        }
        return *this;
    }
    uint64_t node_id;
    uint64_t edge_id;
    uint64_t *first_out;
    uint64_t *head;
  };
  EdgeIterator begin() const {
    return EdgeIterator {first_node(), first_out[first_node()], first_out, head};
  }
  EdgeIterator end() const {
    return EdgeIterator{last_node(), first_out[last_node()], first_out, head};
  }

  uint64_t first_node() const {
    return mpi_rank * (first_out_size - 1 + mpi_size - 1) / mpi_size;
  }
  uint64_t last_node() const {
    auto next = (mpi_rank + 1) * (first_out_size - 1 + mpi_size - 1) / mpi_size;
    return std::min(next, first_out_size - 1);
  }

  size_t size() const { 
    return first_out[last_node()] - first_out[first_node()];
  }

  size_t global_node_count() const {
    return first_out_size - 1;
  }

 private:
  int       mpi_rank, mpi_size;
  int       first_out_fd;
  int head_fd;
  uint64_t* first_out;
  size_t first_out_size;
  uint64_t* head;
  size_t head_size;
};


using graph_type = delegate_partitioned_graph<std::allocator<std::byte>>;

typedef double                         edge_data_type;

using edge_data_allocator_type = std::allocator<edge_data_type>;

graph_type read_graph(std::string const& input_filename, Config const& conf) {
  int mpi_rank = comm_world().rank();
  int mpi_size = comm_world().size();

  if (mpi_rank == 0) {
    std::cout << "MPI initialized with " << mpi_size << " ranks." << std::endl;
  }
  comm_world().barrier();

  // parse_cmd_line(argc, argv, output_filename, backup_filename,
  //                delegate_threshold, input_filenames, gbyte_per_rank,
  //                partition_passes, chunk_size, undirected);
  // typedef delegate_partitioned_graph<distributed_db::allocator<std::byte>> graph_type;

  if (mpi_rank == 0) {
    std::cout << "Ingesting graph from " << input_filename << std::endl;
  }

  // distributed_db ddb(db_create(), output_filename.c_str());
  std::allocator<std::byte> alloc;

  // auto edge_data_ptr =
  //     ddb.get_manager()
  //         ->construct<
  //             graph_type::edge_data<edge_data_type, edge_data_allocator_type>>(
  //             "graph_edge_data_obj")(ddb.get_allocator());

  // Setup edge list reader
  // havoqgt::parallel_edge_list_reader<edge_data_type> pelr(inputs,
                                                          // conf.undirected);
  // bool has_edge_data = pelr.has_edge_data();
  if (conf.gen != Generator::none) {
    std::vector<std::tuple<uint64_t, uint64_t, double>> edge_list;
    std::vector<uint64_t>                               vtxdist;
    kagen::KaGen kagen(MPI_COMM_WORLD);
    kagen.EnableBasicStatistics();
    unsigned long long               n = 1ull << conf.gen_n;
    unsigned long long               m = 1ull << conf.gen_m;
    kagen::KaGenResult result;
    switch(conf.gen) {
      case Generator::rgg_2d:
        result = kagen.GenerateRGG2D_NM(n, m);
        break;
      case Generator::rhg:
      result = kagen.GenerateRHG_NM(conf.gen_gamma, n, m);
        break;
      case Generator::rdg_2d:
        result = kagen.GenerateRDG2D(n, false);
        break;
      case Generator::gnm:
        result = kagen.GenerateUndirectedGNM(n, m);
        break;
      case Generator::rmat:
        result = kagen.GenerateRMAT(n, m, 0.57, 0.19, 0.19);
        break;
      case Generator::none:
        break;
    }
    vtxdist = kagen::BuildVertexDistribution<uint64_t>(result, MPI_UINT64_T, MPI_COMM_WORLD);

    edge_list.reserve(result.edges.size());
    for (const auto& [tail, head] : result.edges) {
      edge_list.emplace_back(tail, head, 0.0);
    }
    auto graph = graph_type(alloc, MPI_COMM_WORLD, edge_list,
                       vtxdist.back() - 1, conf.delegate_threshold,
                       conf.partition_passes, conf.chunk_size);
    return graph;
  } else {
    auto                   inputs = std::vector{{input_filename}};
    BinaryGraphEdgeAdaptor adap(input_filename, MPI_COMM_WORLD);
    auto graph = graph_type(alloc, MPI_COMM_WORLD, adap,
                       adap.global_node_count() - 1, conf.delegate_threshold,
                       conf.partition_passes, conf.chunk_size);
    return graph;
  }

  // auto graph = graph_type(alloc, MPI_COMM_WORLD, pelr, pelr.max_vertex_id(),
  //                         conf.delegate_threshold, conf.partition_passes, conf.chunk_size,
  //                         edge_data);
  // graph_type* graph = ddb.get_manager()->construct<graph_type>("graph_obj")(
  //     ddb.get_allocator(), MPI_COMM_WORLD, pelr, pelr.max_vertex_id(),
  //     delegate_threshold, partition_passes, chunk_size, *edge_data_ptr);

  // if (!has_edge_data) {
  //   ddb.get_manager()
  //       ->destroy<
  //           graph_type::edge_data<edge_data_type, edge_data_allocator_type>>(
  //           "graph_edge_data_obj");
  // }
}

}  // namespace havoqgt::extension