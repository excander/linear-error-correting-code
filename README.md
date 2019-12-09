# linear-error-correting-code
реализация схемы декодирования на основе стандартного расположения (декодирование по синдрому)

### usage: lecc.py [-h] {generator,encoder,decoder} ...

Linear error-correcting code.

optional arguments:
-h, --help            show this help message and exit

mode:
  Valid modes

  {generator,encoder,decoder}
                        
    generator           generating linear code and decode vector.
    
    encoder             encode message m and adding error vector -e.
    
    decoder             decode message and subtracting error vector


### usage: lecc.py generator [-h] [--out-file OUT] r n t

positional arguments:

  r               number of check bits
  
  n               length of a code word
  
  t               number of errors to correct

optional arguments:

  -h, --help      show this help message and exit
  
  --out-file OUT  file to write data for coder/decoder (default: code.data)


### usage: lecc.py encoder [-h] [-e E] inputfile m

positional arguments:

  inputfile   file with encoder data in pickle format
  
  m           message

optional arguments:
  -h, --help  show this help message and exit
  
  -e E        error

### usage: lecc.py decoder [-h] inputfile y

positional arguments:

  inputfile   file with decoder data in pickle format
  
  y           message with error

