#SEGMENTATION MODEL

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.database_setup import Base, Jobs
import datetime
import pickle
import time

engine = create_engine('postgres+psycopg2://postgres:root@localhost:5433/pyvinci')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def database_insert():
    newJobs = list()
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/bikes.jpeg')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/beach.jpeg')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/buildings.JPG')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/dog.jpeg')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/snow_statue.JPG')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/trees_buildings.JPG')
    newJobs.append('https://storage.googleapis.com/segmentation-testing/testing_images1/bike.jpeg')
    for imageURL in newJobs:
        job_new = Jobs(image_url=imageURL, status="PENDING", created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        session.add(job_new)
        session.commit()
        print('New job added: {}'.format(job_new))


def database_delete():
    for record in session.query(Jobs).filter_by(status="COMPLETE").all():
        session.delete(record)
        session.commit()
        print("Record deleted: {}".format(record))


def database_read():
    jobIDs = list()
    imageURLs = list()
    limit_value = 3
    for job_id, image_url in session.query(Jobs.id, Jobs.image_url).filter_by(status="PENDING").order_by(Jobs.created_at).limit(limit_value).all():
        jobIDs.append(job_id)
        imageURLs.append(image_url)
    return (jobIDs, imageURLs)

def database_update(job_id, image_url, labels_things_pred, labels_stuff_pred, masks_labels_pred, masks_nparr_pred):
    job_completed = session.query(Jobs).filter_by(id=job_id).first()
    job_completed.labels_things = labels_things_pred
    job_completed.labels_stuff = labels_stuff_pred
    job_completed.mask_labels = masks_labels_pred
    job_completed.masks_nparr = pickle.dumps(masks_nparr_pred)
    job_completed.status = 'COMPLETE'
    job_completed.updated_at = datetime.datetime.now()
    session.add(job_completed)
    session.commit()
    print('Job completed:{}'.format(job_completed))


def worker():
    jobIDs, imagesURLs = database_read()
    model = SegmentationModel()
    predictor = model.builtModel(UsingCPU=True)
    image_preprocessing = ImagePreProcessing()
    
    for img in imagesURLs:
        image_preprocessing.loadImage(img)
    
    imageProcessedData = image_preprocessing.getImages()

    for i in range(len(jobIDs)):
        image_url, image = imageProcessedData[i]
        prediction = model.getPrediction(predictor, image)
        labels_things_pred, labels_stuff_pred = model.getLabels_PanopticSeg(prediction)
        masks_nparr_pred, masks_labels_pred = model.getMasks_InstanceSeg(prediction, labels=True)
        database_update(jobIDs[i], image_url, labels_things_pred, labels_stuff_pred, masks_labels_pred, masks_nparr_pred)


if __name__ == "__main__":
    while True:
        #worker()
        print('Starting to wait')
        time.sleep(7)
        #main()
        #TESTING
        #database_insert()
        #database_delete()