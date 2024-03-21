from rdflib import Graph
import re
import random


def capture_code(text):
    code_pattern = r"```(?:sparql|SPARQL)?(.*?)```"
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"


# Function to replace prefixed names with their full URIs
def replace_prefixed_names_with_uris(query):
    if not re.search("PREFIX", query):
        return query
    # Extract prefixes and their URIs
    prefix_pattern = r"\s*PREFIX\s*?([a-zA-Z0-9_\-]+):\s*?<([^>]+)>"
    prefixes = dict(re.findall(prefix_pattern, query))

    # Pattern to match prefixed names (simple version, might need refinement)
    prefixed_name_pattern = r'\b(' + '|'.join(
        prefixes.keys()) + r'):([a-zA-Z0-9_\-]+)'

    # Function to replace match with full URI
    def repl(match):
        prefix, local_part = match.groups()
        return f"<{prefixes[prefix]}{local_part}>"

    # Replace all occurrences of prefixed names in the query
    query = re.sub(".*PREFIX.*\n+", "", query)
    return re.sub(prefixed_name_pattern, repl, query)


def extract_identifiers(resolved_query):
    uri_pattern = r'<(http[s]?://[^ >]*dbpedia\.org[^ >]*)>'

    # Use regex to find all URIs in the SPARQL query
    uris = re.findall(uri_pattern, resolved_query, re.I)

    return set(uris)


def parse_ontology(ontology_path):
    name = ontology_path.split("/")[-1].split("_")[0]
    return name


def count_common_elements(list_of_sets):
    num_sets = len(list_of_sets)
    common_elements_count = [0] * num_sets

    for i in range(num_sets):
        for j in range(num_sets):
            if i != j:  # Skip comparing the set with itself
                common_elements_count[i] += len(
                    list_of_sets[i].intersection(list_of_sets[j]))
    return common_elements_count


def get_max_indexes(lst):
    if not lst:
        return []  # Return an empty list if input list is empty
    max_val = max(lst)
    return [i for i, val in enumerate(lst) if val == max_val]


def self_consistency(queries_list, ontology):

    parsed_code = [replace_prefixed_names_with_uris(query) for query in
                   queries_list]
    extracted_identifiers = [extract_identifiers(code) for code in parsed_code]

    agreement = count_common_elements(extracted_identifiers)
    max_index = get_max_indexes(agreement)
    if max_index != 1:
        candidates = [extracted_identifiers[idx] for idx in max_index]
        consistency = [len(query_uri & ontology) for query_uri in candidates]
        max_index = get_max_indexes(consistency)
        if max_index != 1:
            return queries_list[random.choice(max_index)]
    return queries_list[max_index[-1]]

def self_consistency(queries_list, ontology):

    parsed_code = [replace_prefixed_names_with_uris(query) for query in
                   queries_list]
    extracted_identifiers = [extract_identifiers(code) for code in parsed_code]

    agreement = count_common_elements(extracted_identifiers)
    print(agreement)
    max_index = get_max_indexes(agreement)
    print(max_index)
    if max_index != 1:
        candidates = [extracted_identifiers[idx] for idx in max_index]
        consistency = [len(query_uri & ontology) for query_uri in candidates]
        _max_index = get_max_indexes(consistency)
        print(_max_index)
        if _max_index != 1:
            print([max_index[idx] for idx in _max_index])
            return queries_list[max_index[random.choice(_max_index)]]
    return queries_list[max_index[-1]]