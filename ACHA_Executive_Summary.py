from main import extract_tables


def process_acha_report():
    pdf_path = "ACHA_WELL-BEING_ASSESSMENT_FALL_2023_REFERENCE_GROUP_EXECUTIVE_SUMMARY.pdf"
    output_folder = "ACHA_Tables"
    extract_tables(pdf_path, output_folder)


if __name__ == "__main__":
    process_acha_report()