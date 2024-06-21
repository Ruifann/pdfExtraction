from main import extract_tables


def process_ncha_report():
    pdf_path = "NCHA-IIIb_FALL_2023_REFERENCE_GROUP_EXECUTIVE_SUMMARY_03.19.24.pdf"
    output_folder = "NCHA_Tables"
    extract_tables(pdf_path, output_folder)


if __name__ == "__main__":
    process_ncha_report()
