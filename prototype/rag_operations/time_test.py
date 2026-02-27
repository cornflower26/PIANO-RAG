import time
import numpy as np
import pandas as pd
import re
import faiss
import json
import sys
import os
import torch
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse
from pathlib import Path

from prototype.fhe_query_client import FHEQueryClient
from prototype.fhe_query_server import FHEQueryServer
from prototype.go_runner import run_built_executable
from prototype.rag_utils import PromptedBGE


def run_pir_rag_experiment(queries: List[np.ndarray], dataset_size: int = 65000, k_clusters: int = 5,
                           cluster_top_k: int = 3, top_k: int = 10) -> Dict[str, Any]:
    """Run PIR-RAG experiment with detailed timing."""
    dataset_info = f" on {dataset_size}"
    print(f"Running PIR-RAG experiment{dataset_info} (k_clusters={k_clusters}, cluster_top_k={cluster_top_k})")
    actual_dataset_size = dataset_size
    if dataset_size > 9999999:
        actual_dataset_size = 1120486
    elif dataset_size > 999999:
        actual_dataset_size = 702873

    # Setup phase
    setup_start = time.perf_counter()

    print("=" * 70)
    print("FHE Query Encryption Client")
    print("=" * 70)
    print()


    # Initialize client
    client = FHEQueryClient(
        context_path=Path("./fhe_context"),
        poly_modulus_degree=8192
    )

    # Initialize server
    server = FHEQueryServer(
        context_path=Path("./fhe_context"),
        centroids_path=Path(f"./prototype/data/65000_centroids.npy"),
        #centroids_path=Path(f"./prototype/data/1000_centroids.npy"),
        poly_modulus_degree=8192
    )



    #client = PIRRAGClient()
    #server = PIRRAGServer()

    # Setup timing breakdown
    cluster_start = time.perf_counter()


    # Server does clustering and setup first
    #server_setup_result = server.setup(embeddings, documents, k_clusters)
    encrypted_query, encrypted_norm, metadata = client.process_query(queries[0])

    # Save encrypted query
    print(f"\n{'=' * 70}")
    print("Saving Encrypted Query")
    print(f"{'=' * 70}")
    output_path = Path("./encrypted_queries")
    query_file, norm_file, metadata_file = client.save_encrypted_query(
        encrypted_query,
        encrypted_norm,
        metadata,
        output_path
    )
    """
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Query: {query[:100] if len(query) > 100 else query}")
    print(f"Query dimension: {metadata['query_dim']}")
    print(f"Plaintext ||q||^2: {metadata['plaintext_norm_squared']:.4f}")
    print(f"Plaintext ||q||: {metadata['plaintext_norm']:.4f}")
    print(f"\nOutput files:")
    print(f"  - Encrypted query: {query_file}")
    print(f"  - Encrypted norm: {norm_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"{'=' * 70}")"""
    clustering_time = time.perf_counter() - cluster_start

    server_setup_start = time.perf_counter()


    # Client setup with centroids from server
    #client_setup_result = client.setup(server.centroids)

    print("=" * 70)
    print("FHE Query Server - Distance Computation")
    print("=" * 70)
    print("Encrypted query upload " + str(sys.getsizeof(encrypted_query)))

    # Load encrypted query
    encrypted_query, encrypted_norm = server.load_encrypted_query(
        Path("./encrypted_queries/encrypted_query.bin"),
        Path("./encrypted_queries/encrypted_norm_squared.bin")
    )

    # Compute distances
    encrypted_distances = server.compute_distances(
        encrypted_query,
        encrypted_norm,
        # batch_size=args.batch_size
    )

    # Save encrypted distances
    server.save_encrypted_distances(
        encrypted_distances,
        Path("./encrypted_distances")
    )
    print("Encrypted distances upload " + str(sys.getsizeof(encrypted_distances)))

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Computed {len(encrypted_distances)} encrypted distances")
    print(f"Output directory: ./encrypted_distances")
    print(f"{'=' * 70}")
    server_setup_time = time.perf_counter() - server_setup_start

    total_setup_time = time.perf_counter() - setup_start

    # Query phase - detailed timing for each step
    query_times = []
    wacky_query_times = []
    step_times = []
    communication_costs = []

    for i, query_embedding in enumerate(queries):
        print(f"  Query {i + 1}/{len(queries)}")

        query_start = time.perf_counter()


        # Decrypt distances
        top_k_distances, top_k_indices = client.decrypt_distances(
            Path("./encrypted_distances"),
            top_k=cluster_top_k
        )

        # Save results
        client.save_decrypted_results(
            top_k_distances,
            top_k_indices,
            Path("./decrypted_results")
        )

        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"Decrypted {len(top_k_distances)} top distances")
        print(f"Results saved to: ./decrypted_results")
        print(f"{'=' * 70}")

        # Step 1: Find relevant clusters
        """
        query_tensor = torch.tensor(query_embedding) if not isinstance(query_embedding,
                                                                       torch.Tensor) else query_embedding
        relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=cluster_top_k)"""

        print(f"\n{'=' * 70}")
        print(f"Step 4: Global top-{top_k} selection...")

        query_vector = queries[0]

        # run executable
        # retrieve info from it

        cluster_start = time.perf_counter()
        stdout, stderr, code = run_built_executable(
            #"/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
            "/home/ajanusze/Piano-PIR-RAG/executables-5rD4OHJASP/___go_build_easypir_client_new_linux",
            #"/home/ajanusze/Piano-PIR-RAG/executables-cbO36P80R8/___go_build_easypir_client_new_linux",
            args=["-ip", "localhost:50052", "-thread", "1", "-input",
                  #"/Users/antoniajanuszewicz/PycharmProjects/
                  "/home/ajanusze/PIANO-RAG/decrypted_results/top_k_results.json",
                  "-extra_input",
                  #f"/Users/antoniajanuszewicz/PycharmProjects/
                  f"/home/ajanusze/PIANO-RAG/prototype/data/{dataset_size}_lists.json", "-numEntries", f"{actual_dataset_size}"],
                  #f"/home/ajanusze/PIANO-RAG/prototype/data/nq_100_lists.json", "-numEntries", f"{actual_dataset_size}"],
            timeout=60
        )
        #print(stderr)
        cluster_time = time.perf_counter() - cluster_start
        metrics = parse_all_metrics(stderr)
        print(metrics)
        _, indices, vectors = extract_query_results(stderr)
        print("Cluster associated indices")
        print(indices)


        top_k_results = pir_global_top_k(
            indices,
            query_vector,
            vectors,
            top_k=top_k,
            use_faiss=True
        )
        top_k_indices = [idx for idx, _ in top_k_results]
        top_k_distances = [dist for _, dist in top_k_results]

        save_top_k_results(np.array(top_k_distances), np.array(top_k_indices),
                           Path("/home/ajanusze/PIANO-RAG/prototype/ground_truth"))

        print(f"✓ Selected top-{len(top_k_results)} vectors:")
        for i, (idx, dist) in enumerate(top_k_results, 1):
            print(f"  {i}. Vector {idx}: distance = {dist:.4f}")

        # Step 2: PIR retrieval (now returns URLs and embeddings together)
        pir_start = time.perf_counter()
        print(f"\n{'=' * 70}")
        print("Step 5: Mapping vector IDs to documents...")
        # ====================================================================
        # DOCUMENT/PICKLE INTERACTION: Getting actual document content
        # ====================================================================
        # - Translates vector indices → Document objects via docstore
        # - Documents contain page_content (text) and metadata (title)
        # documents = get_documents_by_indices(faiss_vectorstore, top_k_indices)
        rerank_start = time.perf_counter()
        documents, metrics2 = pir_get_documents_by_indices(top_k_indices,actual_dataset_size)
        rerank_time = time.perf_counter() - rerank_start
        print(documents)
        print(top_k_indices)
        print(f"✓ Retrieved {len(documents)} documents")
        #doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
        pir_time = time.perf_counter() - pir_start


        # Step 3: Reranking (using embeddings from PIR, no server request)
        #final_results = client.rerank_documents(query_tensor, doc_tuples, top_k=top_k)

        total_query_time = time.perf_counter() - query_start

        query_times.append(total_query_time)
        print(metrics)
        step_times.append({
            'cluster_selection_time': cluster_time,
            'pir_retrieval_time': pir_time,
            'reranking_time': rerank_time,
            'query_time': metrics['end_to_end_amortized_time_ms']*metrics['num_queries']+(metrics2['end_to_end_amortized_time_ms']*metrics2['num_queries']),
            'server_time': metrics['average_server_time_ms']*metrics['num_queries']+(metrics2['average_server_time_ms']*metrics2['num_queries'])
            #'decode_time': pir_metrics.get('total_decode_time', 0)
        })
        communication_costs.append({
            'upload_bytes': metrics['per_query_upload_cost_kb']*metrics['num_queries']+(metrics2['per_query_upload_cost_kb']*metrics2['num_queries']),
            'download_bytes': metrics['per_query_download_cost_kb']*metrics['num_queries']+(metrics2['per_query_download_cost_kb']*metrics2['num_queries'])
        })
        wacky_query_times.append(total_query_time - float(metrics['setup_time'])/1000)

    return {
        'system': 'PIANO-RAG',
        'dataset_size': dataset_size,
        'setup_time': total_setup_time,
        'clustering_time': clustering_time,
        'server_setup_time': server_setup_time,
        'avg_query_time': np.mean(query_times),
        'std_query_time': np.std(query_times),
        'query_times': query_times,
        'step_times': step_times,
        'avg_upload_bytes': np.mean([c['upload_bytes'] for c in communication_costs]),
        'avg_download_bytes': np.mean([c['download_bytes'] for c in communication_costs]),
        'communication_costs': communication_costs,
        'parameters': {'k_clusters': k_clusters, 'cluster_top_k': cluster_top_k, 'top_k': top_k},
        'embedding_dim': 768,
        'end-to-end': query_times[0] + total_setup_time,
        'end to end minus server': total_setup_time + wacky_query_times[0],

    }

def save_top_k_results(
            top_k_distances: np.ndarray,
            top_k_indices: np.ndarray,
            output_path: Path,
            centroids_path: Path = None
    ):
        """
        Save decrypted top-k results to files.

        Args:
            top_k_distances: Top-k distances (sorted)
            top_k_indices: Top-k centroid indices (sorted)
            output_path: Output directory
            centroids_path: Optional path to 65000_centroids.npy for additional info
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save distances and indices
        results = {
            "top_k": len(top_k_distances),
            "distances": top_k_distances.tolist(),
            "centroid_indices": top_k_indices.tolist(),
            "min_distance": float(top_k_distances.min()),
            "max_distance": float(top_k_distances.max()),
            "mean_distance": float(top_k_distances.mean())
        }

        results_file = output_path / "ground_truth.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {results_file}")

        # Save as numpy arrays for easy loading
        # np.save(output_path / "top_k_distances.npy", top_k_distances)
        # np.save(output_path / "top_k_indices.npy", top_k_indices)
        # print(f"✓ Saved numpy arrays to {output_path}")

        # Print summary
        print(f"\nTop-{len(top_k_distances)} Results:")
        print(f"  Best match: Centroid {top_k_indices[0]} (distance: {top_k_distances[0]:.4f})")
        print(f"  Worst match: Centroid {top_k_indices[-1]} (distance: {top_k_distances[-1]:.4f})")

        return results_file

def pir_get_documents_by_indices(
        vector_indices: List[int],dataset_size: int
):
    stdout, stderr, code = run_built_executable(
        #"/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
        "/home/ajanusze/Piano-PIR-RAG/executables-5rD4OHJASP/___go_build_easypir_client_new_linux",
        #"/home/ajanusze/Piano-PIR-RAG/executables-cbO36P80R8/___go_build_easypir_client_new_linux",
        args=["-ip", "localhost:50051", "-thread", "1", "-input",
              #"/Users/atoniajanuszewicz/PycharmProjects
              "/home/ajanusze/PIANO-RAG/prototype/ground_truth/ground_truth.json", "-numEntries", f"{dataset_size}"],
        timeout=60
    )
    print(stderr)
    metrics = parse_all_metrics(stderr)
    print(metrics)
    indices, text = extract_text_query_results(stderr)
    documents = text

    # ====================================================================
    # DOCUMENT/PICKLE INTERACTION: Accessing docstore
    # ====================================================================
    # - docstore contains the actual document text and metadata
    # - Loaded from index.pkl file (pickle format)
    # - docstore._dict maps doc_id → Document object
    return documents,metrics

def extract_text_query_results(log_string):
    """
    Extract indices and text results from query result log strings.

    Parameters:
    -----------
    log_string : str
        The log string containing text query results

    Returns:
    --------
    tuple: (indices, texts)
        indices: list of int - the extracted indices
        texts: list of str - the corresponding text results
    """
    indices = []
    texts = []

    # Split into lines
    lines = log_string.strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for lines that contain "Final query result at index"
        if "Final query result at index" in line:
            # Extract the index number
            match = re.search(r'at index (\d+):', line)
            if match:
                index = int(match.group(1))

                # Extract the text after the colon
                # Split at ': ' and take everything after
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    text = parts[1].strip()

                    indices.append(index)
                    texts.append(text)

        i += 1

    return indices, texts

def extract_query_results(log_string):
    """
    Extract cluster IDs, indices, and vector arrays from query result log strings.

    Parameters:
    -----------
    log_string : str
        The log string containing query results with cluster information

    Returns:
    --------
    tuple: (cluster_ids, indices, vectors)
        cluster_ids: list of int - the cluster IDs for each query result
        indices: list of int - the extracted indices
        vectors: np.ndarray - array of shape (n, vector_dim) containing the vectors
    """
    cluster_ids = []
    indices = []
    vectors = []

    # Split into lines
    lines = log_string.strip().split('\n')

    for line in lines:
        # Look for lines that contain "Final query result from cluster"
        if "Final query result from cluster" in line:
            # Extract the cluster ID and index
            # Pattern: "from cluster <cluster_id> at index <index>:"
            match = re.search(r'from cluster (\d+) at index (\d+):', line)
            if match:
                cluster_id = int(match.group(1))
                index = int(match.group(2))

                # Extract the vector (everything inside the square brackets)
                vector_match = re.search(r'\[(.*?)\]', line)
                if vector_match:
                    vector_str = vector_match.group(1)
                    # Split by whitespace and convert to floats
                    vector = np.array([float(x) for x in vector_str.split()])

                    cluster_ids.append(cluster_id)
                    indices.append(index)
                    vectors.append(vector)

    # Convert list of vectors to numpy array
    if vectors:
        vectors = np.array(vectors)
    else:
        vectors = np.array([])

    return cluster_ids, indices, vectors

def pir_global_top_k(
        indexes: List[str],
        query_vector: np.ndarray,
        all_vectors: np.ndarray,
        top_k: int = 10,
        use_faiss: bool = True
) -> List[Tuple[int, float]]:
    """
    Merge all candidates and do a global top-k sort.

    Can use pure NumPy (default) or FAISS IndexFlatL2 for KNN.

    Args:
        candidates: List of (vector_index, distance) tuples
        query_vector: Query embedding vector
        all_vectors: All document vectors
        top_k: Number of top results to return
        use_faiss: Whether to use FAISS IndexFlatL2 (vs pure NumPy)

    Returns:
        Top-k (vector_index, distance) tuples sorted by distance
    """
    if use_faiss and len(indexes) > 0:
        # ====================================================================
        # FAISS INTERACTION: Building temporary FAISS IndexFlatL2 for KNN
        # ====================================================================
        # - Creates a small FAISS index on candidate vectors
        # - Uses FAISS search() for efficient distance computation
        # - Alternative to pure NumPy sorting (faster for large candidate sets)
        candidate_indices = indexes
        candidate_vectors = all_vectors

        # Create FAISS index
        dim = candidate_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(candidate_vectors.astype('float32'))

        # Query
        query_reshaped = query_vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_reshaped, k=top_k)

        # Map back to original indices
        results = [
            (candidate_indices[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        return results
    return [(0,0)]


def parse_all_metrics(text) -> Dict[str, float]:
    """
    Parse all available metrics from the log output.

    Returns more comprehensive metrics dictionary.
    """
    metrics = {}

    # Basic metrics
    patterns = {
        'setup_time': r'Setup Phase took (\d+) ms',
        'num_queries': r'Finish Online Phase with (\d+) queries',
        'per_query_upload_cost_kb': r'Per query upload cost ([\d.]+) kb',
        'per_query_download_cost_kb': r'Per query download cost ([\d.]+) kb',
        'end_to_end_amortized_time_ms': r'End to end amortized time ([\d.]+) ms',
        'average_client_time_ms': r'Average Client Time ([\d.]+) ms',
        'average_server_time_ms': r'Average Server Time ([\d.]+) ms',
        'average_online_time_ms': r'Average Online Time ([\d.]+) ms',
        'average_network_latency_ms': r'Average Network Latency ([\d.]+) ms',
        'average_find_hint_time_ms': r'Average Find Hint Time ([\d.]+) ms',
        'local_storage_size_mb': r'Local Storage Size ([\d.]+) MB',
        'setup_phase_time_ms': r'Setup Phase took (\d+) ms',
        'online_phase_time_ms': r'Online Phase took (\d+) ms',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if 'num' in key or 'queries' in key:
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)

    return metrics

def embed_query(query_text: str) -> np.ndarray:
    """Embed query → vector q."""
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    query_vec = embeddings.embed_query(query_text)
    return np.array(query_vec, dtype=np.float32)

def main():
    #to allow this to run, this runs with the bigger databases (this shouldn't affect the numbers?)
    query = "What is machine learning?"
    embedded_query = embed_query(query)
    """
    query_path = Path("/home/ajanusze/PIANO-RAG/nq_100_queries.json")
    query_vector = parse_questions_from_file(query_path)
    print(query_vector)
    queries = []
    # len(query_vector)
    for i in range(0, len(query_vector)):
        queries.append(np.array(embed_query(query_vector[i][1])))
        print(query_vector[i][1])
    """

    datasets = [1000, 5000, 10000, 65000,1000000]
    #datasets = [65000]
    clusters = [50,250,500,4096,4096]
    cluster_top = [4,20,40,100,100]
    test = 0

    #print(run_pir_rag_experiment([embedded_query], dataset_size=datasets[test], k_clusters=clusters[test],cluster_top_k=cluster_top[test], top_k=10))
    print(run_pir_rag_experiment([embedded_query], dataset_size=65000, k_clusters=4096, cluster_top_k=50, top_k=50))


    #with open("./nq_results.txt", 'a') as f:
    #    for q in queries:
    #        print("Next one")
    #        f.write(str(run_pir_rag_experiment([q], dataset_size=datasets[test], k_clusters=clusters[test],cluster_top_k=cluster_top[test], top_k=10)))
    #        f.write("\n")

    return



def parse_questions_from_file(filepath):
    """
    Parse question data from a JSON file and return a dictionary mapping IDs to questions.

    Args:
        filepath: Path to the file containing question data (newline-delimited JSON format)

    Returns:
        dict: Dictionary mapping id (int) -> question (str)
    """
    id_to_question = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = [(item['query_id'], item['query_text']) for item in data['queries']]
    return results

if __name__ == "__main__":
    main()
