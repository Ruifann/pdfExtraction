import tabula
import pandas as pd
import logging


def extract_tables(pdf_path):

    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    logging.info(f"Number of tables extracted: {len(tables)}")
    return tables


def save_tables(tables):
    for index, table in enumerate(tables):
        csv_path = f"Institution_Information_table_{index + 1}.csv"
        table.to_csv(csv_path, index=False)
        logging.info(f"Table {index + 1} saved to {csv_path}")


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    logging.debug(df.head())
    if df.isnull().sum().any():
        df_dropped = df.dropna()
        df_dropped.to_csv(csv_path, index=False)
        logging.info(f"Data cleaned and saved to {csv_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    pdf_path = "Institution_Information.pdf"
    tables = extract_tables(pdf_path)
    if tables:
        save_tables(tables)
        first_table_csv = 'Institution_Information_table_1.csv'
        load_and_clean_data(first_table_csv)


if __name__ == "__main__":
    main()
