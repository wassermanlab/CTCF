import sys

#++++++++++++++++#
# I/O Utilities  #
#++++++++++++++++#

def __file_handle(file_name, mode="r"):
    """
    @input:
    file_name {str}
    mode {str} e.g. "r", rb", etc.; default = "r"

    @yield: {str}
    """

    raiseValueError = False
    
    # Open a file handle
    if file_name.endswith(".gz"):
        try:
            import gzip
            fh = gzip.open(file_name, mode)
        except:
            raiseValueError = True

    elif file_name.endswith(".zip"):
        try:
            from zipfile import ZipFile
            zf = ZipFile(file_name, mode)
            for f in zf.infolist():
                # i.e. only handles the first file
                fh = zf.open(f, mode)
                break
        except:
            raiseValueError = True

    else:
        try:
            fh = open(file_name, mode)
        except:
            raiseValueError = True
    
    if raiseValueError:
        raise ValueError("Could not open file handle: %s" % file_name)

    return(fh)

def parse_file(file_name):
    """
    Parses a file and yields lines one by one.

    @input:
    file_name {str}

    @yield: {str}
    """

    fh = __file_handle(file_name)

    # For each line...
    for line in fh:
        yield(line.strip("\n"))

    fh.close()

def parse_csv_file(file_name, sep=","):
    """
    Parses a CSV file and yields lines one by one as a list.

    @input:
    file_name {str}
    sep {str} e.g. "\t"; default = ","

    @yield: {list}
    """

    import pandas as pd

    fh = __file_handle(file_name)

    # Read in chunks
    for chunk in pd.read_csv(
        fh, header=None, encoding="utf8", sep=sep, chunksize=1024, comment="#"
    ):
        for index, row in chunk.iterrows():
            yield(row.tolist())

    fh.close()

def parse_tsv_file(file_name):
    """
    Parses a TSV file and yields lines one by one as a list.

    @input:
    file_name {str}

    @yield: {list}
    """

    # For each line...
    for line in parse_csv_file(file_name, sep="\t"):
        yield(line)

def parse_json_file(file_name):

    import json

    return(json.loads("\n".join([l for l in parse_file(file_name)])))

def parse_fasta_file(file_name):
    """
    Parses a FASTA file and yields {SeqRecord} objects one by one.

    @input:
    file_name {str}

    @yield: {SeqRecord}
    """

    from Bio import SeqIO

    fh = __file_handle(file_name, mode="rt")

    # For each SeqRecord...
    for seq_record in SeqIO.parse(fh, "fasta"):
        yield(seq_record)

    fh.close()

def write(file_name=None, content=None, overwrite=False):
    """
    Writes content to a file. If overwrite=False, content will be appended at
    the end of the file. If file_name=None, writes content to STDOUT. 

    @input:
    file_name {str}
    content {str}
    overwrite {bool}
    """

    if file_name:
        if overwrite:
            mode = "w"
        else:
            mode = "a"
        fh = __file_handle(file_name, mode=mode)
        fh.write("%s\n" % content)
        fh.close()

    else:
        sys.stdout.write("%s\n" % content)
