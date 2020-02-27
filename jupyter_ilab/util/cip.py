import xarray as xa

CreateIPServer = "https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/CREATE-IP/"

CIP_addresses = {
    "merra2": CreateIPServer + "/reanalysis/MERRA2/mon/atmos/{}.ncml",
    "merra": CreateIPServer + "/reanalysis/MERRA/mon/atmos/{}.ncml",
    "ecmwf": CreateIPServer + "/reanalysis/ECMWF/mon/atmos/{}.ncml",
    "cfsr": CreateIPServer + "/reanalysis/CFSR/mon/atmos/{}.ncml",
    "20crv": CreateIPServer + "/reanalysis/20CRv2c/mon/atmos/{}.ncml",
    "jra": CreateIPServer + "/reanalysis/JMA/JRA-55/mon/atmos/{}.ncml",
}

class CIP:

    @classmethod
    def address(cls,  model: str, varName: str) -> str:
        return CIP_addresses[model.lower()].format(varName)

    @classmethod
    def data_array(cls,  model: str, varName: str) -> xa.DataArray:
        data_address = cls.address( model, varName )
        dataset = xa.open_dataset(data_address)
        return dataset[ varName ]