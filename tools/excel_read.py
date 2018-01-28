from openpyxl import Workbook,load_workbook

excel_file = load_workbook('123.xlsx')
sheet_name = excel_file.get_sheet_names()
print(sheet_name)