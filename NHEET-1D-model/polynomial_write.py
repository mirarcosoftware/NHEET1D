class Polynomial:
    def __init__(self, coefficients,string1,string2,numtype):
        self.coeffs = coefficients
        self.string1 = string1
        self.string2 = string2
        self.numtype = numtype

    def __str__(self):
        chunks = []
        for coeff, power in zip(self.coeffs, range(0,len(self.coeffs))): #range(len(self.coeffs) - 1, -1, -1))
            if coeff == 0:
                continue
            chunks.append(self.format_coeff(coeff,self))
            chunks.append(self.format_power(power,self))
            if power != len(self.coeffs)-1:
                chunks.append('+')
                #print(power)
        chunks[0] = '%s=' % self.string2 +chunks[0].lstrip("+") 
        #print(chunks)
        return ''.join(chunks)

    @staticmethod
    def format_coeff(coeff,self):
        #return str(coeff) if coeff < 0 else "+{0}".format(coeff)
        #return str('%.7e'%coeff) if coeff < 0 else "+{0}".format('%.7e'%coeff)
        #return str(self.numtype%coeff) if coeff < 0 else "+{0}".format(self.numtype%coeff)
        return str(coeff)

    @staticmethod
    def format_power(power,self):
        if power ==0:
            tmp = ''
        elif power ==1:
            #tmp = '%s' % self.string1
            tmp = '*%s' % self.string1
        else:
            #tmp = '%s^{0}'.format(power) % self.string1
            tmp = '*%s^{0}'.format(power) % self.string1
        return tmp
        #return '%s^{0}'.format(power) % self.string1 if power > 1 else ''

#####
import numpy as np
class Polynomial2:
    def __init__(self, coefficients,string1,string2,string3,numtype):
        self.coeffs = coefficients
        self.string1 = string1
        self.string2 = string2
        self.string3 = string3
        self.numtype = numtype
        

    def __str__(self):
        chunks = []
        
#         order=None
#         kx = int(len(self.coeffs)/2-1)
#         ky=kx
#         cofs = np.ones((kx+1, ky+1))
#         print(np.ndindex(cofs.shape))
#         power1 = [0]*len(self.coeffs)
#         power2 = [0]*len(self.coeffs)
#         print(cofs)
#         for index, (j, i) in enumerate(np.ndindex(cofs.shape)):
#             # do not include powers greater than order
#             #print(i,j)
#             if order is not None and i + j > order:
#                 #arr = np.zeros_like(x)
#                 power1[i] = 0
#                 power2[j] = 0
#             else:
#                 #arr = cofs[i, j] * x**i * y**j
#                 #a[index] = arr.ravel()
#                 power1[i] = i
#                 power2[j] = j
#         print(power1,power2)
        
        inc = 0
        for coeff,power in zip(self.coeffs, range(0,len(self.coeffs))): #range(len(self.coeffs) - 1, -1, -1))
            i2 = int(np.floor(inc/3))
            if inc ==0:
                power1 = 0
                power2 = 0
            elif inc ==1:
                power1 = 1
                power2 = 0
            elif inc == 2:
                power1 = 0
                power2 = 1
            elif inc%3 ==0:
                power1 = i2+1
                power2 = i2-1
            elif inc%3==1:
                power1 = i2
                power2 = i2
            elif inc%3 ==2:
                power1 = i2-1
                power2 = i2+1
            #print(power1,power2)
            if coeff == 0:
                continue
            chunks.append(self.format_coeff(coeff,self))
            chunks.append(self.format_power(power1,self.string1))
            chunks.append(self.format_power(power2,self.string2))
            if inc != len(self.coeffs)-1:
                chunks.append('+')
                #print(power)
            inc = inc+1
        #chunks[0] = '%s=' % self.string3 +chunks[0].lstrip("+") 
        #print(chunks)
        return ''.join(chunks)

    @staticmethod
    def format_coeff(coeff,self):
        #return str(coeff) if coeff < 0 else "+{0}".format(coeff)
        #return str('%.7e'%coeff) if coeff < 0 else "+{0}".format('%.7e'%coeff)
        #return str(self.numtype%coeff) if coeff < 0 else "+{0}".format(self.numtype%coeff)
        return str(coeff)

    @staticmethod
    def format_power(power,self):
        if power ==0:
            tmp = ''
        elif power ==1:
            #tmp = '%s' % self.string1
            tmp = '*%s' % self
        else:
            #tmp = '%s^{0}'.format(power) % self.string1
            tmp = '*%s^{0}'.format(power) % self
        return tmp
        #return '%s^{0}'.format(power) % self.string1 if power > 1 else ''