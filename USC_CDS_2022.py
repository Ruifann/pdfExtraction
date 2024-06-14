from main import extract_tables


def process_usc_cds_2022():
    pdf_path = "USC_CDS_2022-2023.pdf"
    output_folder = "USC_CDS_2022_Tables"
    extract_tables(pdf_path, output_folder)


if __name__ == "__main__":
    process_usc_cds_2022()
