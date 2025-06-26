def main():
    with open('dir_pdfs.txt', 'r', encoding='utf-8') as f:
        dir_files = set(line.strip() for line in f if line.strip())
    with open('db_pdfs.txt', 'r', encoding='utf-8') as f:
        db_files = set(line.strip() for line in f if line.strip())
    missing = dir_files - db_files
    print(f'Files in directory but not in DB: {len(missing)}')
    print('Sample missing:', list(missing)[:10])

if __name__ == "__main__":
    main() 