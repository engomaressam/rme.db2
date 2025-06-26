import pandas as pd
import os
import re

# Read the Excel file
excel_path = 'YouTube_Video_List.xlsx'
df = pd.read_excel(excel_path)

# Build the folder tree from titles
folder_tree = {}
for _, row in df.iterrows():
    # Split on ' - ', '\\', or '/'
    parts = re.split(r' - |\\|/', str(row['Title']))
    parts = [p.strip() for p in parts if p.strip()]
    url = row['URL']
    node = folder_tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node.setdefault('__videos__', []).append((parts[-1], url))

# Recursive function to generate HTML for the tree
def render_tree(node, level=0):
    html = ''
    indent = '  ' * level
    for key, value in node.items():
        if key == '__videos__':
            for title, url in value:
                html += f'{indent}<li class="video"><a href="{url}" target="_blank">{title}</a></li>\n'
        else:
            html += f'{indent}<li><span class="folder" onclick="toggleFolder(this)">📁 {key}</span>\n'
            html += f'{indent}<ul class="nested">' + render_tree(value, level+1) + f'{indent}</ul></li>\n'
    return html

html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YouTube Video Catalog</title>
<style>
body {{ font-family: Arial, sans-serif; }}
ul {{ list-style-type: none; }}
.folder {{ cursor: pointer; font-weight: bold; }}
.nested {{ display: none; margin-left: 20px; }}
.active {{ display: block; }}
.video a {{ text-decoration: none; color: #1a0dab; }}
.video a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h2>YouTube Video Catalog</h2>
<ul id="tree">
{render_tree(folder_tree)}
</ul>
<script>
function toggleFolder(element) {{
  var nextUl = element.nextElementSibling;
  if (nextUl) {{
    nextUl.classList.toggle('active');
  }}
}}
// Expand the first level by default
document.addEventListener('DOMContentLoaded', function() {{
  var folders = document.querySelectorAll('#tree > li > .folder');
  folders.forEach(function(folder) {{
    var ul = folder.nextElementSibling;
    if (ul) ul.classList.add('active');
  }});
}});
</script>
</body>
</html>
'''

with open('youtube_catalog.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('✅ youtube_catalog.html created. Open it in your browser!') 