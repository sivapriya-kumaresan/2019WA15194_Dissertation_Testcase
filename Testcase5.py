import xml.etree.ElementTree as ET

xml_string = "<root><element1>Value 1</element1><element2>Value 2</element2>"

# Parse XML string
root = ET.fromstring(xml_string)

