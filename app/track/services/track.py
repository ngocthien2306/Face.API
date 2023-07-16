from datetime import datetime
from sqlalchemy.orm import Session
from app.track import models
from app.track.models.track import Track
from sqlalchemy import or_, select, and_
from app.track.models.vehicle import Vehicle
from app.track.services.face import FaceServices
from app.track.enums.track import VehicleStatus
from api.track.v1.request.track import CheckVehicleRequest
from core.db import Transactional, session
import cv2
import os
import base64
import numpy as np
from sqlalchemy import text
from sqlalchemy.future import select
face_services = FaceServices()
class TrackingServices:
    def __init__(self, PATH_IMAGE_SAVE=None):
        self.path_image_save = PATH_IMAGE_SAVE
    async def checkidVehicle(self, plate_number: str) -> Vehicle:
        print(plate_number)
        query = select(models.Vehicle).where(Vehicle.plateNum == plate_number)
        vehicle = await session.execute(query)
        
        return vehicle.scalars().first()
    async def checkidVehicleInParking(self, vehicleId: int) -> Track:
        # Sort with Time
        query = select(models.Track).where( and_(
            Track.vehicleId == vehicleId,
            Track.detectOutFace == "0",
            Track.plateOut == "0"
        ))
        trackVehicle = await session.execute(query)
        return trackVehicle.scalars().first()
    @Transactional()
    async def create_track_vehicle(self,model,img_detected_save, plate_number: str,imglp_detected: str,typeTransaction: str, typeLP: str, img_detected: str):
        ## Verify that the Plate Number
        try:
                print(plate_number)
                ## Create the Vehicle model
                # Check if the face car has arrived in the parking lot 
                ## Query vehicle in database
                vehicleCheck = await self.checkidVehicle(plate_number)
                if not vehicleCheck :
                    vehicle = Vehicle(
                        plateNum=plate_number,
                        status = VehicleStatus.ACCEPTIN.value,
                        typeTransport = typeTransaction,
                        typePlate = typeLP
                    )
                    session.add(vehicle)
                    
                    session.flush()
                    session.refresh(vehicle)
                    vehicleCheck= vehicle
                track = await self.checkidVehicleInParking(vehicleCheck.id)
                print("track",track)
                curDT = datetime.now()
                if not track:
                    trackVehicle = Track(
                        vehicleId = vehicleCheck.id,
                        trackNumber = 1,
                        startTime = curDT,
                        fee = 0.0,
                        siteId =1,
                        detectInFace = img_detected,
                        plateIn = imglp_detected
                    )
                    session.add(trackVehicle)
                else:
                    ## Face verify
                    img_detected_path = track.detectInFace
                    print(img_detected_path)
                    # Reshape
                    img_detected_save= cv2.resize(img_detected_save,(112,112))
                    access =face_services.face_check_track(model,img_detected_save,img_detected_path)
                    print(access)
                    if access == True:
                        track.endTime =curDT
                        track.detectOutFace = img_detected
                        track.plateOut = imglp_detected
                        vehicleCheck.status= VehicleStatus.ACCEPTOUT.value
                        session.refresh(vehicleCheck)
                    else:
                        vehicleCheck.status =VehicleStatus.BLOCK.value
                        session.refresh(vehicleCheck)
                session.commit()

                return {"status": str(vehicleCheck.status),"fee": 0}
        except Exception as e:
            return {"status": "Error is:"+ str(e),"fee": 0}  
    @Transactional()
    async def create_track_vehicle_async(self, 
                                        model,
                                        img_detected_save,
                                        imglp_detected: str,
                                        img_detected: str,
                                        request: CheckVehicleRequest
                                        ):
        ## Verify that the Plate Number
        try:
            print(request.platenum)
            result = await session.execute(
                text
                (
                    '''
                        EXEC TRACK_MANAGEMENT 
                            @Method=:Method, 
                            @PlateNum=:PlateNum, 
                            @StatusVehicle=:StatusVehicle, 
                            @TypeTransport=:TypeTransport, 
                            @TypePlate=:TypePlate,
                            @SiteId=:SiteId,
                            @TrackNumber=:TrackNumber,
                            @Fee=:Fee,
                            @DetectInFace=:DetectInFace,
                            @PlateIn=:PlateIn
                    '''
                )
                .params(
                        Method="SaveTrack", 
                        PlateNum=request.platenum, 
                        StatusVehicle=VehicleStatus.ACCEPTIN.value, 
                        TypeTransport=request.typeTransport, 
                        TypePlate=request.typeLicensePlate,
                        SiteId=request.siteId,
                        TrackNumber=1,
                        Fee=0,
                        DetectInFace=img_detected,
                        PlateIn=imglp_detected
                    )
            )
            results = result.fetchone()
            print(results)
            is_exist = results.isExistYN
            if is_exist == 'Y':
                # Face verify
                detectInFace = results.faceIn
                trackId = results.trackId
                vehicleId = results.vehicleId
                print()
                img_detected_path = detectInFace
                print(img_detected_path)
                # Reshape
                img_detected_save = cv2.resize(img_detected_save,(112,112))
                access =face_services.face_check_track(model,img_detected_save,img_detected_path)
                print(access)
                if access: 
                    status = VehicleStatus.ACCEPTOUT.value
                else: 
                    status = VehicleStatus.BLOCK.value
                result = await session.execute(
                    text
                    (
                        '''
                            EXEC TRACK_MANAGEMENT 
                                @Method=:Method, 
                                @DetectOutFace=:DetectOutFace,
                                @PlateOut=:PlateOut,
                                @StatusVehicle=:StatusVehicle,
                                @TrackId=:TrackId,
                                @VehicleId=:VehicleId
                        '''
                    )
                    .params(
                            Method="UpdateStatusVehicle", 
                            DetectOutFace=img_detected,
                            PlateOut=imglp_detected,
                            StatusVehicle=status, 
                            TrackId=trackId,
                            VehicleId=vehicleId
                        )
                    )
                    
                results = result.fetchone()
                

                    
            return {"status": str(results.statusVehicle),"fee": 0}
        except Exception as e:
            return {"status": "Error is:"+ str(e),"fee": 0}  
    def convertbase64 (self,string64 ):
        decoded_data = base64.b64decode(string64)
        np_data = np.fromstring(decoded_data, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return image
    @Transactional()
    async def create_track_report (self,siteid,platenumber,typeVehicle,typeLp,img_detected, imglp_detected) -> bool:
        print(siteid,platenumber,typeVehicle,typeLp)
        # Save Image data
        
        result = await session.execute(
                    text
                    (
                        ''' 
                            DECLARE @ErrorMessage NVARCHAR(500);
                            EXEC SP_UPDATE_TRACK_REPORT 
                                @Platenum=:PlateNumber,
                                @TypeTransport=:TypeTransport,
                                @TypePlate=:TypePlate,
                                @SiteId=:SiteId,
                                @DetectOutFace=:DetectOutFace,
                                @PlateOut=:PlateOut,
                                @Status=:StatusVehicle,
                                @ErrorMessage = @ErrorMessage OUTPUT;
                            SELECT @ErrorMessage AS ErrorMessage;
                        '''
                    )
                    .params(
                            PlateNumber = platenumber,
                            TypeTransport = typeVehicle,
                            TypePlate = typeLp,
                            SiteId = siteid,
                            DetectOutFace=img_detected,
                            PlateOut=imglp_detected,
                            StatusVehicle=VehicleStatus.REPORT.value, 

                        )
                )
                  
        error = result.fetchone()
        print("Result",type(error))
        print("Result",error.ErrorMessage)
        if error.ErrorMessage is None:
            return True
        # else rollback
        else:
            if os.path.exists(imglp_detected):
                os.remove(imglp_detected)
            if os.path.exists(imglp_detected):
                os.remove(imglp_detected)
            return False