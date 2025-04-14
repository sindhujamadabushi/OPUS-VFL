import re
import os

# Function to extract arrays from each file
def extract_terms(file_path, org_index):
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Extract contribution_term using regex
        contribution_match = re.search(r'contribution_term= \[(.*?)\]', content)
        privacy_match = re.search(r'privacy_term= \[(.*?)\]', content)
        
        if contribution_match and privacy_match:
            contribution_term = list(map(float, contribution_match.group(1).split(',')))
            privacy_term = list(map(float, privacy_match.group(1).split(',')))
            
            # Return specific element based on org_index
            return contribution_term[org_index], privacy_term[org_index]
        else:
            raise ValueError(f"Arrays not found in {file_path}")

# Main function to process all files
def main(org_index, directory='.'):  # Default directory is current
    result_ct_array = []
    result_pt_array = []

    for i in range(1, 21):  # Assuming files numbered from 1 to 20
        file_name = f"slurm_out_2942992_{i}.log"
        file_path = os.path.join(directory, file_name)

        try:
            ct, pt = extract_terms(file_path, org_index)
            result_ct_array.append(ct)
            result_pt_array.append(pt)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return result_ct_array, result_pt_array

# Example usage
if __name__ == "__main__":
    org = 2
    directory = "logs_ab_sweep_mnist/"
    
    result_ct_array, result_pt_array = main(org, directory)
    print("contribution_terms=", result_ct_array)
    print("privacy_terms=", result_pt_array)
