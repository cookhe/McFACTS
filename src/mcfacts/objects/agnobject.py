import numpy as np
from mcfacts.setup import setupdiskstars

def dump_array_to_file(fname=None, samples_out=None):
  # Write output (ascii)
  # Vastly superior data i/o with pandas, but note no # used
  import pandas
  dframe = pandas.DataFrame(samples_out)
  fname_out_ascii= fname
  dframe.to_csv(fname_out_ascii,sep=' ',header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],index=False) # # is not pre-appended...just boolean

""" 
def check_1d_length(arr_list):
    if(len({arr.shape for arr in arr_list}) > 1):
        raise ValueError('Arrays are not all 1d.')

    if( len(set(map(len,arr_list))) != 1):
        raise ValueError('Arrays are not all the same length.')
 """    

#TODO: method to print values when you just run an AGNOBject class in a jupyter notebook or similar.
#TODO: similar vein: print method? Same thing maybe?
#TODO: custom error messages when you don't supply all the fields for init and add methods
#TODO: custom error messages when input arrays to init and add methods aren't the same length
#TODO: dump_record_array writes every value as a float. Make dictionary with all attributes and datatypes? Or is a float fine?
#TODO: init_from_file: you have to initialize an empty AGNObject before you can init_from_file and that seems weird.
       #check if this is now just an AGNObject because we would want it to be an AGNStar or AGNBlackHole etc
#TODO: error if you try to pass kwarg that doesnt exist. AttributeError?
#TODO: issue: right now you can pass AGNStar arrays that are 8-elements long for the star parameters and 10-elements long 
       #for the AGNObject parameters and it doesn't complain.
#TODO: dump_array_to_file: can I just import DataFrame and to_csv so it's lower overhead?

class AGNObject(object):
    """
    An array of objects. Should include all objects of this type.
    Argument arrays should be *1d scalar arrays*.
    Everything here is for the center-of-mass around the SMBH
    """

    def __init__(self,  mass = None,
                        spin = None, #internal quantity. total J for a binary
                        spin_angle = None, #angle between J and orbit around SMBH for binary
                        orbit_a = None, #location
                        orbit_inclination= None, #of CoM for binary around SMBH
                        orb_ang_mom = None,  # redundant, should be computed from keplerian orbit formula for L in terms of mass, a, eccentricity
                        orbit_e = None,
                        nsystems = None):
        
        #Make sure all inputs are included
        if mass is None: raise AttributeError("mass is not included in inputs")
        if spin is None: raise AttributeError('spin is not included in inputs')
        if spin_angle is None: raise AttributeError('spin_angle is not included in inputs')
        if orbit_a is None: raise AttributeError('orbit_a is not included in inputs')
        if orbit_inclination is None: raise AttributeError('orbit_inclination is not included in inputs')
#        if orb_ang_mom is None: raise AttributeError('orb_ang_mom is not included in inputs')
        if orbit_e is None: raise AttributeError('orbit_e is not included in inputs')

        if nsystems is None: nsystems = mass.size

        assert mass.shape == (nsystems,),"mass: all arrays must be 1d and the same length"
        assert spin.shape == (nsystems,),"spin: all arrays must be 1d and the same length"
        assert spin_angle.shape == (nsystems,),"spin_angle: all arrays must be 1d and the same length"
        assert orbit_a.shape == (nsystems,),"orbit_a: all arrays must be 1d and the same length"
        assert orbit_inclination.shape == (nsystems,),"orbit_inclination: all arrays must be 1d and the same length"
#        assert orb_ang_mom.shape == (nsystems,),"orb_ang_mom: all arrays must be 1d and the same length"
        assert orbit_e.shape == (nsystems,),"orbit_e: all arrays must be 1d and the same length"
        
        self.mass = mass #Should be array. TOTAL masses.
        self.spin = spin #Should be array
        self.spin_angle = spin_angle #should be array
        self.orbit_a = orbit_a #Should be array. Semimajor axis
        self.orbit_inclination = orbit_inclination #Should be array. Allows for misaligned orbits.
        self.orb_ang_mom = orb_ang_mom #needs to be added in!
        self.orbit_e = orbit_e #Should be array. Allows for eccentricity.
        self.generations = np.full(nsystems,1)
    
    def __add_objects__(self, new_mass = None,
                              new_spin = None,
                              new_spin_angle = None,
                              new_a = None,
                              new_inclination = None,
#                              new_orb_ang_mom = None,
                              new_e = None,
                              nsystems = None):
        """
        Adds new values to the end of existing arrays
        """

        #Make sure all inputs are included
        if new_mass is None: raise AttributeError('new_mass is not included in inputs')
        if new_spin is None: raise AttributeError('new_spin is not included in inputs')
        if new_spin_angle is None: raise AttributeError('new_spin_angle is not included in inputs')
        if new_a is None: raise AttributeError('new_a is not included in inputs')
        if new_inclination is None: raise AttributeError('new_inclination is not included in inputs')
#        if new_orb_ang_mom is None: raise AttributeError('new_orb_ang_mom is not included in inputs')
        if new_e is None: raise AttributeError('new_e is not included in inputs')

        if nsystems is None: nsystems = new_mass.size

        assert new_mass.shape == (nsystems,),"new mass: all arrays must be 1d and the same length"
        assert new_spin.shape == (nsystems,),"new_spin: all arrays must be 1d and the same length"
        assert new_spin_angle.shape == (nsystems,),"new_spin_angle: all arrays must be 1d and the same length"
        assert new_a.shape == (nsystems,),"new_a: all arrays must be 1d and the same length"
        assert new_inclination.shape == (nsystems,),"new_inclination: all arrays must be 1d and the same length"
#        assert new_orb_ang_mom.shape == (nsystems,),"new_orb_ang_mom: all arrays must be 1d and the same length"
        assert new_e.shape == (nsystems,),"new_e: all arrays must be 1d and the same length"


        self.mass = np.concatenate([self.mass,new_mass])
        self.spin = np.concatenate([self.spin,new_spin])
        self.spin_angle = np.concatenate([self.spin_angle,new_spin_angle])
        self.orbit_a = np.concatenate([self.orbit_a,new_a])
        self.orbit_inclination = np.concatenate([self.orbit_inclination,new_inclination])
#        self.orb_ang_mom = np.concatenate([self.orb_ang_mom,new_orb_ang_mom])
        self.orbit_e = np.concatenate([self.orbit_e,new_e])
        self.generations = np.concatenate([self.generations,np.full(len(new_mass),1)])

    def remove_objects(self, idx_remove = None):
        """
        Removes objects at specified indices.

        Parameters
        ----------
        idx_remove : numpy array
            Indices to remove

        Returns
        -------
        ???
        idx_remove should be a numpy array of indices to change, e.g., [2, 15, 23]
        as written, this loops over all attributes, not just those in the AGNObjects class,
        i.e., we don't need a separate remove_objects method for the subclasses.
        """

        #Check that the index array is a numpy array.
        assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_remove is None:
            return None
        
        idx_change = np.ones(len(self.mass),dtype=bool)
        idx_change[idx_remove] = False
        for attr in vars(self).keys():
            setattr(self,attr,getattr(self,attr)[idx_change])

    def locate(self, idx=None):
        """
        Returns objects at specified indices
        """

        #Check that index array is numpy array
        assert isinstance(idx,np.ndarray),"idx must be numpy array"

        if idx is None:
            return None
        
        idx_full = np.zeros(len(self.mass),dtype=bool)
        idx_full[idx] = True

    def sort(self, sort_attr=None):
        #Takes in one attribute array and sorts the whole class by that array
        #Sorted array indices
        sort_idx = np.argsort(sort_attr)

        #Now apply these to each of the attributes
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[sort_idx])

    def return_params(self):
        """
        Gets list of parameters present in object.

        Parameters
        ----------
        None

        Returns
        -------
        list
            parameters in object
        """
        return(list(vars(self).keys()))


    def return_record_array(self,**kwargs):
        """
        Right now every dtype is float.
        Loops over all attributes, don't need to rewrite for subclasses.
        """
        dtype = np.dtype([(attr,'float') for attr in vars(self).keys()])
        dat_out = np.empty(len(self.mass),dtype=dtype)
        for attr in vars(self).keys():
            dat_out[attr] = getattr(self,attr)
        return(dat_out)
    
    def to_file(self,fname=None):
        # Write output (ascii)
        # Vastly superior data i/o with pandas, but note no # used
        import pandas
        samples_out = self.return_record_array()
        dframe = pandas.DataFrame(samples_out)
        fname_out_ascii= fname
        dframe.to_csv(fname_out_ascii,sep=' ',header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],index=False) # # is not pre-appended...just boolean

    
    def init_from_file(self,fname=None):
        """
        Read in previously saved data file and create AGNObject from it.
        Odd implementation right now, have to init AGNObject and only then read from file.
        """
        dat_in = np.genfromtxt(fname,names=True)
        for name in dat_in.dtype.names:
            setattr(self,name,dat_in[name])


class AGNStar(AGNObject):
    """
    An array of single stars. Should include all objects of this type. No other objects should contain objects of this type.
    Takes in initial helium and metals mass fractions, calculates hydrogen mass fraction from that to ensure it adds up to unity.
    """
    def __init__(self, star_radius = None, star_Y = None, star_Z = None, n_stars = None, **kwargs):
        """
        Array of single star objects.
        star_radius should be a 1d numpy array
        star_Y and star_Z can be floats or 1d numpy arrays, but must sum to 1 or less.
        """
        #Make sure all inputs are included
        #if star_radius is None: raise AttributeError('star_radius is not included in inputs')
        if star_Y is None: raise AttributeError('star_Y is not included in inputs')
        if star_Z is None: raise AttributeError('star_Z is not included in inputs')

        if n_stars is None: n_stars = star_radius.size

        #Make sure inputs are numpy arrays
        if star_radius is None:
            self.star_radius = setupdiskstars.setup_disk_stars_radii(masses=mass)

        else:
            assert isinstance(star_radius,np.ndarray),"star_radius is not a numpy array"
            self.star_radius = star_radius

        if (np.any(star_Y + star_Z > 1.)):
            raise ValueError("star_Y and star_Z must sum to 1 or less.")

        if ((isinstance(star_Y,float) and (isinstance(star_Z,float)))):
            self.star_X = np.ones(len(star_radius)) - np.full(len(star_radius),star_Y) - np.full(len(star_radius),star_Z)
            self.star_Y = np.full(len(star_radius),star_Y)
            self.star_Z = np.full(len(star_radius),star_Z)
        
        elif ((isinstance(star_Y,np.ndarray) and isinstance(star_Z,np.ndarray))):
            assert star_radius.shape == (n_stars,),"star_radius: all arrays must be 1d and the same length"
            assert star_Y.shape == (n_stars,),"star_Y, array: all arrays must be 1d and the same length"
            assert star_Z.shape == (n_stars,),"star_Z, array: all arrays must be 1d and the same length"

            self.star_X = 1. - star_Y - star_Z
            self.star_Y = star_Y
            self.star_Z = star_Z

        else:
            raise TypeError("star_Y and star_Z must be either both floats or numpy arrays")


        super(AGNStar,self).__init__(nsystems = n_stars, **kwargs) #calls top level functions
    
    def __repr__(self):
        return('AGNStar(): {} single stars'.format(len(self.mass)))

    def add_stars(self, new_radius = None, new_Y = None, new_Z = None, nsystems = None, **kwargs):
        """
        Add new star values to the end of existing arrays.
        """

        #Make sure all inputs are included
        if new_radius is None: raise AttributeError("new_radius is not included in inputs")
        if new_Y is None: raise AttributeError("new_Y is not included in inputs")
        if new_Z is None: raise AttributeError("new_Z is not included in inputs")

        if nsystems is None: nsystems = new_radius.size

        assert new_radius.shape == (nsystems,),"new_radius: all arrays must be 1d and the same length"
        self.star_radius = np.concatenate([self.star_radius,new_radius])

        if(np.any(new_Y + new_Z) > 1.): raise ValueError("new_Y and new_Z must sum to 1 or less")

        if( (isinstance(new_Y, float)) and (isinstance(new_Z, float))):
            self.star_X = np.concatenate(self.star_X, np.full(nsystems,1.-new_Y-new_Z))
            self.star_Y = np.concatenate([self.star_Y, np.full(nsystems,new_Y)])
            self.star_Z = np.concatenate([self.star_Z, np.full(nsystems,new_Z)])
            
        if( (isinstance(new_Y, np.ndarray)) and (isinstance(new_Z, np.ndarray))):
            self.star_X = np.concatenate([self.star_X, np.ones(nsystems) - new_Y - new_Z])
            self.star_Y = np.concatenate([self.star_Y, new_Y])
            self.star_Z = np.concatenate([self.star_Z, new_Z])
        super(AGNStar,self).__add_objects__(nsystems = nsystems, **kwargs)


class AGNBlackHole(AGNObject):
    """
    An array of single black holes. Should include all objects of this type. No other objects should contain objects of this type.
    """
    def __init__(self,**kwargs):
        super(AGNBlackHole,self).__init__(**kwargs) #Calls top level functions
    
    def __repr__(self):
        return('AGNBlackHole(): {} single black holes'.format(len(self.mass)))


    def add_blackholes(self, nsystems = None, **kwargs):
        super(AGNBlackHole,self).__add_objects__(nsystems = nsystems, **kwargs)


class AGNBinaryStar(AGNObject):
    """
    An array of binary stars. Should include all objects of this type. No other objects should include objects of this type.
    Properties of this class:
        * scalar properties of each star:  masses, radius, Y,Z (star_mass1,star_mass2, star_radius1, star_radius2, ...)
         * vector properties of each star: individual angular momentum vectors (spin1,spin_angle_1) *relative to the z axis of the binary orbit*,
           not SMBH
         * orbit properties: use a reduced mass hamiltonian, you have 'r = r2 - r1' (vector) for which we get binary_a, binary_e, binary_inclination (relative to SMBH)

    """

    def __init__(self, star_mass1 = None,
                       star_mass2 = None,
                       star_radius1 = None,
                       star_radius2 = None,
                       star_Y1 = None,
                       star_Y2 = None,
                       star_Z1 = None,
                       star_Z2 = None,
                       star_spin1 = None,
                       star_spin2 = None,
                       star_spin_angle1 = None,
                       star_spin_angle2 = None,
                       binary_e = None,
                       binary_a = None,
                       binary_inclination=None,
                       cm_orbit_a=None,
                       cm_orbit_inclination=None,
                       cm_orbit_e=None,
                       nsystems = None,
                     **kwargs):
        
        #Make sure all inputs are included
        if star_mass1 is None: raise AttributeError("star_mass1 is not included in inputs")
        if star_mass2 is None: raise AttributeError("star_mass2 is not included in inputs")
        if star_radius1 is None: raise AttributeError("star_radius1 is not included in inputs")
        if star_radius2 is None: raise AttributeError("star_radius2 is not included in inputs")
        if star_Y1 is None: raise AttributeError("star_Y1 is not included in inputs")
        if star_Y2 is None: raise AttributeError("star_Y2 is not included in inputs")
        if star_Z1 is None: raise AttributeError("star_Z1 is not included in inputs")
        if star_Z2 is None: raise AttributeError("star_Z2 is not included in inputs")
        if star_spin1 is None: raise AttributeError("star_spin1 is not included in inputs")
        if star_spin2 is None: raise AttributeError("star_spin2 is not included in inputs")
        if star_spin_angle1 is None: raise AttributeError("star_spin_angle1 is not included in inputs")
        if star_spin_angle2 is None: raise AttributeError("star_spin_angle2 is not included in inputs")
        if binary_e is None: raise AttributeError("binary_e is not included in inputs")
        if binary_a is None: raise AttributeError("binary_a is not included in inputs")

        if nsystems is None: nsystems = star_mass1.size

        #Check that all inputs are 1d numpy arrays
        assert star_mass1.shape == (nsystems,),"star_mass1: all arrays must be 1d and the same length"
        assert star_mass2.shape == (nsystems,),"star_mass2: all arrays must be 1d and the same length"
        assert star_radius1.shape == (nsystems,),"star_radius1: all arrays must be 1d and the same length"
        assert star_radius2.shape == (nsystems,),"star_radius2: all arrays must be 1d and the same length"
        assert star_spin1.shape == (nsystems,),"star_spin1: all arrays must be 1d and the same length"
        assert star_spin2.shape == (nsystems,),"star_spin2: all arrays must be 1d and the same length"
        assert star_spin_angle1.shape == (nsystems,),"star_spin_angle1: all arrays must be 1d and the same length"
        assert star_spin_angle2.shape == (nsystems,),"star_spin_angle2: all arrays must be 1d and the same length"
        assert binary_e.shape == (nsystems,),"binary_e: all arrays must be 1d and the same length"
        assert binary_a.shape == (nsystems,),"binary_a: all arrays must be 1d and the same length"

        if (np.any(star_Y1 + star_Z1 > 1.)):
            raise ValueError("star_Y1 and star_Z1 must sum to 1 or less.")
        if (np.any(star_Y2 + star_Z2 > 1.)):
            raise ValueError("star_Y2 and star_Z2 must sum to 1 or less.")

        if((isinstance(star_Y1,float)) and (isinstance(star_Z1,float))):
            star_X1 = np.full(nsystems,1. - star_Y1 - star_Z1)
            star_Y1 = np.full(nsystems,star_Y1)
            star_Z1 = np.full(nsystems,star_Z1)
        if((isinstance(star_Y2,float)) and (isinstance(star_Z2,float))):
            star_X2 = np.full(nsystems,1. - star_Y2 - star_Z2)
            star_Y2 = np.full(nsystems,star_Y2)
            star_Z2 = np.full(nsystems,star_Z2)
        if((isinstance(star_Y1,np.ndarray)) and (isinstance(star_Z1,np.ndarray))):
            assert star_Y1.shape == (nsystems,),"star_Y1: all arrays must be 1d and the same length"
            assert star_Z1.shape == (nsystems,),"star_Z1: all arrays must be 1d and the same length"
            star_X1 = np.ones(nsystems) - star_Y1 - star_Z1
        if((isinstance(star_Y2,np.ndarray)) and (isinstance(star_Z2,np.ndarray))):
            assert star_Y2.shape == (nsystems,),"star_Y2: all arrays must be the same length"
            assert star_Z2.shape == (nsystems,),"star_Z2: all arrays must be the same length"
            star_X2 = np.ones(nsystems) - star_Y2 - star_Z2
        else: raise TypeError("star_Y1, star_Z1 and star_Y2, star_Z2 must be either both floats or numpy arrays")


        #Now assign attributes
        self.star_mass1 = star_mass1
        self.star_mass2 = star_mass2
        self.star_radius1 = star_radius1
        self.star_radius2 = star_radius2
        self.star_X1 = star_X1
        self.star_X2 = star_X2
        self.star_Y1 = star_Y1
        self.star_Y2 = star_Y2
        self.star_Z1 = star_Z1
        self.star_Z2 = star_Z2
        self.star_spin1 = star_spin1
        self.star_spin2 = star_spin2
        self.star_spin_angle1 = star_spin_angle1
        self.star_spin_angle2 = star_spin_angle2
        self.binary_e = binary_e
        self.binary_a = binary_a
        self.binary_inclination = binary_inclination

        #Now calculate properties for the AGNObject class aka the totals
        total_mass = star_mass1 + star_mass2
        total_spin = None # we will pass None for now, until we decide how to treat angular momentum
        total_spin_angle = None  # we will pass None for now, until we decide how to treat angular momentum

        super(AGNBinaryStar,self).__init__(mass = total_mass, spin = star_spin_angle2, spin_angle = star_spin_angle2, orbit_a = cm_orbit_a, orbit_inclination = cm_orbit_inclination, orbit_e = cm_orbit_e, nsystems = nsystems)
        

    def __repr__(self):
        return('AGNBinaryStar(): {} stellar binaries'.format(len(self.mass)))
    
    def add_binaries(self, new_star_mass1 = None,
                        new_star_radius1 = None,
                        new_star_mass2 = None,
                        new_star_radius2 = None,
                        new_star_Y1 = None,
                        new_star_Y2 = None,
                        new_star_Z1 = None,
                        new_star_Z2 = None,
                        new_star_spin1 = None,
                        new_star_spin2 = None,
                        new_star_spin_angle1 = None,
                        new_star_spin_angle2 = None,
                        new_star_orbit_a1 = None,
                        new_star_orbit_a2 = None,
                        new_binary_e = None,
                        new_binary_a = None,
                        new_binary_inclination=None,
                        new_cm_orbit_a=None,
                        new_cm_orbit_inclination=None,
                        new_cm_orbit_e=None,
                        nsystems = None,
                    **kwargs):
        

        #Make sure all inputs are included
        if new_star_mass1 is None: raise AttributeError("new_star_mass1 is not included in inputs")
        if new_star_mass2 is None: raise AttributeError("new_star_mass2 is not included in inputs")
        if new_star_radius1 is None: raise AttributeError("new_star_radius1 is not included in inputs")
        if new_star_radius2 is None: raise AttributeError("new_star_radius2 is not included in inputs")
        if new_star_Y1 is None: raise AttributeError("new_star_Y1 is not included in inputs")
        if new_star_Y2 is None: raise AttributeError("new_star_Y2 is not included in inputs")
        if new_star_Z1 is None: raise AttributeError("new_star_Z1 is not included in inputs")
        if new_star_Z2 is None: raise AttributeError("new_star_Z2 is not included in inputs")
        if new_star_spin1 is None: raise AttributeError("new_star_spin1 is not included in inputs")
        if new_star_spin2 is None: raise AttributeError("new_star_spin2 is not included in inputs")
        if new_star_spin_angle1 is None: raise AttributeError("new_star_spin_angle1 is not included in inputs")
        if new_star_spin_angle2 is None: raise AttributeError("new_star_spin_angle2 is not included in inputs")
        if new_binary_e is None: raise AttributeError("new_binary_e is not included in inputs")
        if new_binary_a is None: raise AttributeError("new_binary_a is not included in inputs")

        if nsystems is None: nsystems = new_star_mass1.size

        #Check that all inputs are 1d numpy arrays
        assert new_star_mass1.shape == (nsystems,),"new_star_mass1: all arrays must be 1d and the same length"
        assert new_star_mass2.shape == (nsystems,),"new_star_mass2: all arrays must be 1d and the same length"
        assert new_star_radius1.shape == (nsystems,),"new_star_radius1: all arrays must be 1d and the same length"
        assert new_star_radius2.shape == (nsystems,),"new_star_radius2: all arrays must be 1d and the same length"
        assert new_star_spin1.shape == (nsystems,),"new_star_spin1: all arrays must be 1d and the same length"
        assert new_star_spin2.shape == (nsystems,),"new_star_spin2: all arrays must be 1d and the same length"
        assert new_star_spin_angle1.shape == (nsystems,),"new_star_spin_angle1: all arrays must be 1d and the same length"
        assert new_star_spin_angle2.shape == (nsystems,),"new_star_spin_angle2: all arrays must be 1d and the same length"
        assert new_binary_e.shape == (nsystems,),"new_binary_e: all arrays must be 1d and the same length"
        assert new_binary_a.shape == (nsystems,),"new_binary_a: all arrays must be 1d and the same length"

        if (np.any(new_star_Y1 + new_star_Z1 > 1.)):
            raise ValueError("new_star_Y1 and new_star_Z1 must sum to 1 or less.")
        if (np.any(new_star_Y2 + new_star_Z2 > 1.)):
            raise ValueError("new_star_Y2 and new_star_Z2 must sum to 1 or less.")

        if((isinstance(new_star_Y1,float)) and (isinstance(new_star_Z1,float))):
            new_star_X1 = np.full(nsystems,1. - new_star_Y1 - new_star_Z1)
            new_star_Y1 = np.full(nsystems,new_star_Y1)
            new_star_Z1 = np.full(nsystems,new_star_Z1)
        if((isinstance(new_star_Y2,float)) and (isinstance(new_star_Z2,float))):
            new_star_X2 = np.full(nsystems,1. - new_star_Y2 - new_star_Z2)
            new_star_Y2 = np.full(nsystems,new_star_Y2)
            new_star_Z2 = np.full(nsystems,new_star_Z2)
        if((isinstance(new_star_Y1,np.ndarray)) and (isinstance(new_star_Z1,np.ndarray))):
            assert new_star_Y1.shape == (nsystems,),"new_star_Y1: all arrays must be 1d and the same length"
            assert new_star_Z1.shape == (nsystems,),"new_star_Z1: all arrays must be 1d and the same length"
            new_star_X1 = np.ones(nsystems) - new_star_Y1 - new_star_Z1
        if((isinstance(new_star_Y2,np.ndarray)) and (isinstance(new_star_Z2,np.ndarray))):
            assert new_star_Y2.shape == (nsystems,),"new_star_Y2: all arrays must be the same length"
            assert new_star_Z2.shape == (nsystems,),"new_star_Z2: all arrays must be the same length"
            new_star_X2 = np.ones(nsystems) - new_star_Y2 - new_star_Z2
        else: raise TypeError("new_star_Y1, new_star_Z1 and new_star_Y2, new_star_Z2 must be either both floats or numpy arrays")


        #Now add new values
        self.star_mass1 = np.concatenate([self.star_mass1, new_star_mass1])
        self.star_mass2 = np.concatenate([self.star_mass2, new_star_mass2])
        self.star_radius1 = np.concatenate([self.star_radius1, new_star_radius1])
        self.star_radius2 = np.concatenate([self.star_radius2, new_star_radius2])
        self.star_X1 = np.concatenate([self.star_X1, new_star_X1])
        self.star_Y1 = np.concatenate([self.star_Y1, new_star_Y1])
        self.star_Z1 = np.concatenate([self.star_Z1, new_star_Z1])
        self.star_X2 = np.concatenate([self.star_X2, new_star_X2])
        self.star_Y2 = np.concatenate([self.star_Y2, new_star_Y2])
        self.star_Z2 = np.concatenate([self.star_Z2, new_star_Z2])
        self.star_spin1 = np.concatenate([self.star_spin1, new_star_spin1])
        self.star_spin2 = np.concatenate([self.star_spin2, new_star_spin2])
        self.star_spin_angle1 = np.concatenate([self.star_spin_angle1, new_star_spin_angle1])
        self.star_spin_angle2 = np.concatenate([self.star_spin_angle2, new_star_spin_angle2])
        self.binary_e = np.concatenate([self.binary_e, new_binary_e])
        self.binary_a = np.concatenate([self.binary_a, new_binary_a])

        new_total_mass = new_star_mass1 + new_star_mass2

        super(AGNBinaryStar,self).__add_objects__(new_mass = new_total_mass,
                                                   new_spin=None,
                                                   new_spin_angle=None,
                                                   new_a = new_cm_orbit_a,
                                                   new_inclination=new_cm_orbit_inclination,
                                                   new_e=new_cm_orbit_e,
                                                   nsystems=nsystems)


        
        






