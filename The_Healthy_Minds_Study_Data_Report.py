from main import extract_tables


def process_hms_report():
    pdf_path = "HMS_National-Report-2022-2023_full.pdf"
    output_folder = "Healthy_Minds_Tables"
    extract_tables(pdf_path, output_folder)


if __name__ == "__main__":
    process_hms_report()
