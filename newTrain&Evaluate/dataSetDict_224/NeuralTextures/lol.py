# Read the content of the file
with open('trainDataSet.txt', 'r') as file:
    content = file.read()

# Replace commas with spaces
modified_content = content.replace(',', ' ')

# Write the modified content back to the file
with open('trainDataSet.txt', 'w') as file:
    file.write(modified_content)
