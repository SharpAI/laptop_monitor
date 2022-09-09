import os
import requests
import uuid
import logging

labelstudio_url = os.environ['LABEL_STUDIO_URL']
labelstudio_token = os.environ['LABEL_STUDIO_TOKEN']

class LabelStudioClient:
    @staticmethod
    def upload_file(file_path,project_id):
        random_uuid4_str = str(uuid.uuid4())
        dst_filename = random_uuid4_str+'.jpg'

        try:
            os.symlink(file_path, dst_filename)
        except FileExistsError as e:
            pass

        filepath = dst_filename

        auth_header = {'Authorization' : 'Token {}'.format(labelstudio_token)}
        param = {'commit_to_project': 'false'}
        files = {'file': open(filepath, 'rb')}
        get_task_url = f"{labelstudio_url}/api/projects/{project_id}/import"

        response = requests.post(get_task_url,files=files, params=param,headers=auth_header)
        os.unlink(dst_filename)
        if response.ok:
            logging.debug("File uploaded!")
            json_response = response.json()
            file_ids = json_response['file_upload_ids']
            file_id = None
            if file_ids is not None and len(file_ids) > 0:
                file_id = file_ids[0]
            if file_id is not None:
                get_upload_file_url = f"{labelstudio_url}/api/import/file-upload/{file_id}"
                response = requests.get(get_upload_file_url,headers=auth_header)
                if response.ok:
                    logging.debug(response.json())
                    return response.json()
                else:
                    logging.error('cant get uploaded info')
        else:
            logging.error("Error uploading file!")
        return None
    @staticmethod
    def create_task_with_file(file_path,project_id):
        resp = LabelStudioClient.upload_file(file_path,project_id)
        if resp != None:
            fileurl = '/data/' + resp['file']
            task_json = [{
                "data": {"image": fileurl},
                "annotations": [],
                "predictions": []
                }
            ]

            auth_header = {'Authorization' : f'Token {labelstudio_token}'}
            task_url = f"{labelstudio_url}/api/projects/{project_id}/import"
            response = requests.post(task_url, json=task_json,  headers=auth_header)
            if response.ok:
                created_task = response.json()
                logging.debug(created_task)
                return True
            else:
                logging.error('Failed to create Task {}'.format(response))


if __name__ == '__main__':
    # resp = LabelStudioClient.upload_file('./img.jpg',1)
    # print(resp)
    
    resp =LabelStudioClient.create_task_with_file('./img.jpg',1)
    print(resp)