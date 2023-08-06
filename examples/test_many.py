import exrex

for i in exrex.generate('\\d{3}[-]\\d{3}[-]\\d{4}'):
    print(i)