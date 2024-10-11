import yaml

def read_credentials_from_yaml(filepath):
    with open(filepath, "r") as file:
        credentials = yaml.safe_load(file)
    return credentials

# Example usage
credentials = read_credentials_from_yaml("Credentials.yml")
username = credentials['Credentials']['username']
password = credentials['Credentials']['password']

print(f"Username: {username}, Password: {password}")

