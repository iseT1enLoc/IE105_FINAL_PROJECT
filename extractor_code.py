import pefile
import csv
import os
def extract_pe_info(file_path):
    pe = pefile.PE(file_path)

    # Extract relevant information
    pe_info = {
        "File Name": file_path,
        "e_magic":pe.DOS_HEADER.e_magic,
        "e_cblp":pe.DOS_HEADER.e_cblp,
        "e_cp":pe.DOS_HEADER.e_cp,
        "e_crlc":pe.DOS_HEADER.e_crlc,
        "e_cparhdr":pe.DOS_HEADER.e_cparhdr,
        "e_minalloc":pe.DOS_HEADER.e_minalloc,
        "e_maxalloc":pe.DOS_HEADER.e_maxalloc,
        "e_ss":pe.DOS_HEADER.e_ss,
        "e_sp":pe.DOS_HEADER.e_sp,
        "e_csum":pe.DOS_HEADER.e_csum,
        "e_ip":pe.DOS_HEADER.e_ip,
        "e_cs":pe.DOS_HEADER.e_cs,
        "e_lfarlc":pe.DOS_HEADER.e_lfarlc,
        "e_ovno":pe.DOS_HEADER.e_ovno,
        "e_oemid":pe.DOS_HEADER.e_oemid,
        "e_oeminfo":pe.DOS_HEADER.e_oeminfo,
        "e_lfanew":pe.DOS_HEADER.e_lfanew,
        
        "Machine": pe.FILE_HEADER.Machine,
        "NumberOfSections":pe.FILE_HEADER.NumberOfSections,
        "TimeDateStamp":pe.FILE_HEADER.TimeDateStamp,
        "PointerToSymbolTable":pe.FILE_HEADER.PointerToSymbolTable,
        "NumberOfSymbols":pe.FILE_HEADER.NumberOfSymbols,
        "SizeOfOptionalHeader":pe.FILE_HEADER.SizeOfOptionalHeader,
        "Characteristics":pe.FILE_HEADER.Characteristics,     
           
        "Magic":pe.OPTIONAL_HEADER.Magic,
        "MajorLinkerVersion": pe.OPTIONAL_HEADER.MajorLinkerVersion,
        "MinorLinkerVersion": pe.OPTIONAL_HEADER.MinorLinkerVersion,
        "SizeOfCode": pe.OPTIONAL_HEADER.SizeOfCode,
        "SizeOfInitializedData": pe.OPTIONAL_HEADER.SizeOfInitializedData,
        "SizeOfUninitializedData": pe.OPTIONAL_HEADER.SizeOfUninitializedData,
        "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
        "BaseOfCode": pe.OPTIONAL_HEADER.BaseOfCode,
        "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
        "SectionAlignment": pe.OPTIONAL_HEADER.SectionAlignment,
        "FileAlignment": pe.OPTIONAL_HEADER.FileAlignment,
        "MajorOperatingSystemVersion": pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
        "MinorOperatingSystemVersion": pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
        "MajorImageVersion": pe.OPTIONAL_HEADER.MajorImageVersion,
        "MinorImageVersion": pe.OPTIONAL_HEADER.MinorImageVersion,
        "MajorSubsystemVersion": pe.OPTIONAL_HEADER.MajorSubsystemVersion,
        "MinorSubsystemVersion": pe.OPTIONAL_HEADER.MinorSubsystemVersion,
        "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
        "CheckSum": pe.OPTIONAL_HEADER.CheckSum,
        "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
        "Subsystem": pe.OPTIONAL_HEADER.Subsystem, 
        "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
        "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
        "SizeOfStackCommit": pe.OPTIONAL_HEADER.SizeOfStackCommit,
        "SizeOfHeapReserve": pe.OPTIONAL_HEADER.SizeOfHeapReserve,
        "SizeOfHeapCommit": pe.OPTIONAL_HEADER.SizeOfHeapCommit,
        "LoaderFlags": pe.OPTIONAL_HEADER.LoaderFlags,
        "NumberOfRvaAndSizes": pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
        "Subsytem":pe.OPTIONAL_HEADER.Subsystem,
        
        "ImageDirectoryEntryExport":pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].VirtualAddress,
        "ImageDirectoryEntryImport": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].VirtualAddress,
        "ImageDirectoryEntryResource":pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].VirtualAddress,
        "ImageDirectoryEntryException": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']].VirtualAddress,
        "ImageDirectoryEntrySecurity":pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']].VirtualAddress,
        
        "DirectoryEntryImport": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].VirtualAddress,
        "DirectoryEntryImportSize": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size,
        "DirectoryEntryExport": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].VirtualAddress,
        "ImageDirectoryEntryExport": pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size,
    
    }
    print(len(pe_info))
    return pe_info


    

def save_to_csv(folder_path,output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["File Name","e_magic","e_cblp","e_cp","e_crlc","e_cparhdr","e_minalloc","e_maxalloc","e_ss","e_sp","e_csum", "e_csum","e_ip","e_cs","e_lfarlc","e_ovno","e_oemid","e_oeminfo","e_lfanew"
                      ,"Machine","NumberOfSections","TimeDateStamp","PointerToSymbolTable","NumberOfSymbols","SizeOfOptionalHeader","Characteristics",
                      "Magic","MajorLinkerVersion","MinorLinkerVersion","SizeOfCode","SizeOfInitializedData","SizeOfUninitializedData","AddressOfEntryPoint","BaseOfCode","ImageBase","SectionAlignment","FileAlignment","MajorOperatingSystemVersion","MinorOperatingSystemVersion","MajorImageVersion","MinorImageVersion","MajorSubsystemVersion","MinorSubsystemVersion","SizeOfHeaders","CheckSum","SizeOfImage","Subsystem",  "DllCharacteristics", "SizeOfStackReserve","SizeOfStackCommit","SizeOfHeapReserve","SizeOfHeapCommit","LoaderFlags","NumberOfRvaAndSizes","Subsystem"
                      "ImageDirectoryEntryExport","ImageDirectoryEntryImport" ,"ImageDirectoryEntryResource","ImageDirectoryEntryException","ImageDirectoryEntrySecurity", 'ImageDirectoryEntrySecurity', 'DirectoryEntryImport',
                      "DirectoryEntryImport","DirectoryEntryImportSize","DirectoryEntryExport","ImageDirectoryEntryExport"
                     ]
        print(len(fieldnames))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
    
        # Loop through each file in the directory
        for filename in os.listdir(folder_path):
            # Check if the file is a PE file
            if filename.endswith(".exe") or filename.endswith(".dll") or filename.endswith(".cpl") or  filename.endswith(".tsp"):
                # Construct the full path to the PE file
                file_path = os.path.join(folder_path, filename)
                
                try:
                    pe_info = extract_pe_info(file_path)

                    # Write the information to the CSV file
                    writer.writerow({
                        "File Name": pe_info["File Name"],
                        "e_magic":pe_info["e_magic"],
                        "e_cblp":pe_info["e_cblp"],
                        "e_cp":pe_info["e_cp"],
                        "e_crlc":pe_info["e_crlc"],
                        "e_cparhdr":pe_info["e_cparhdr"],
                        "e_minalloc":pe_info["e_minalloc"],
                        "e_maxalloc":pe_info["e_maxalloc"],
                        "e_ss":pe_info["e_ss"],
                        "e_sp":pe_info["e_sp"],
                        "e_csum":pe_info["e_csum"],
                        "e_ip":pe_info["e_ip"],
                        "e_cs":pe_info["e_cs"],
                        "e_lfarlc":pe_info["e_lfarlc"],
                        "e_ovno":pe_info["e_ovno"],
                        "e_oemid":pe_info["e_oemid"],
                        "e_oeminfo":pe_info["e_oeminfo"],
                        "e_lfanew":pe_info["e_lfanew"],
                        "Machine": pe_info["Machine"],
                        "Magic": pe_info["Magic"],
                        "MajorLinkerVersion": pe_info["MajorLinkerVersion"],
                        "MinorLinkerVersion": pe_info["MinorLinkerVersion"],
                        "SizeOfCode": pe_info["SizeOfCode"],
                        "SizeOfInitializedData": pe_info["SizeOfInitializedData"],
                        "SizeOfUninitializedData": pe_info["SizeOfUninitializedData"],
                        "AddressOfEntryPoint": pe_info["AddressOfEntryPoint"],
                        "BaseOfCode": pe_info["BaseOfCode"],
                        "ImageBase": pe_info["ImageBase"],
                        "SectionAlignment": pe_info["SectionAlignment"],
                        "FileAlignment": pe_info["FileAlignment"],
                        "MajorOperatingSystemVersion": pe_info["MajorOperatingSystemVersion"],
                        "MinorOperatingSystemVersion": pe_info["MinorOperatingSystemVersion"],
                        "MajorImageVersion": pe_info["MajorImageVersion"],
                        "MinorImageVersion": pe_info["MinorImageVersion"],
                        "MajorSubsystemVersion": pe_info["MajorSubsystemVersion"],
                        "MinorSubsystemVersion": pe_info["MinorSubsystemVersion"],
                        "SizeOfHeaders": pe_info["SizeOfHeaders"],
                        "CheckSum": pe_info["CheckSum"],
                        "SizeOfImage": pe_info["SizeOfImage"],
                        "Subsystem": pe_info["Subsystem"], 
                        "DllCharacteristics": pe_info["DllCharacteristics"],
                        "SizeOfStackReserve": pe_info["SizeOfStackReserve"],
                        "SizeOfStackCommit": pe_info["SizeOfStackCommit"],
                        "SizeOfHeapReserve": pe_info["SizeOfHeapReserve"],
                        "SizeOfHeapCommit": pe_info["SizeOfHeapCommit"],
                        "LoaderFlags": pe_info["LoaderFlags"],
                        "NumberOfRvaAndSizes": pe_info["NumberOfRvaAndSizes"],
                        "Subsystem":pe_info["Subsystem"],
                        
                        "ImageDirectoryEntryExport":pe_info["ImageDirectoryEntryExport"],
                        "ImageDirectoryEntryImport":pe_info[ "ImageDirectoryEntryImport"] ,
                        "ImageDirectoryEntryResource":pe_info["ImageDirectoryEntryResource"],
                        "ImageDirectoryEntryException":pe_info["ImageDirectoryEntryException"],
                        "ImageDirectoryEntrySecurity":pe_info["ImageDirectoryEntrySecurity"],

                        "DirectoryEntryImport": pe_info["DirectoryEntryImport"],
                        "DirectoryEntryImportSize":pe_info["DirectoryEntryImportSize"] ,
                        "DirectoryEntryExport":pe_info["DirectoryEntryExport"] ,
                        "ImageDirectoryEntryExport":pe_info["ImageDirectoryEntryExport"],
                        
                        
                        
                        "NumberOfSections":pe_info["NumberOfSections"],
                        "TimeDateStamp":pe_info["TimeDateStamp"],
                        "PointerToSymbolTable":pe_info[ "PointerToSymbolTable"],
                        "NumberOfSymbols":pe_info["NumberOfSymbols"],
                        "SizeOfOptionalHeader":pe_info["SizeOfOptionalHeader"],
                        "Characteristics":pe_info["Characteristics"],
                    })

                    

                except pefile.PEFormatError as e:
                    print(f"Error reading {filename}: {e}")

            else:
                print(f"{filename} is not a PE file")


      
def main():
    save_to_csv("D:\\features extractor\\pe_folder","test_extractor.csv") 

if __name__ == "__main__":
    main()





