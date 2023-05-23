//
//  CustomCameraView.swift
//  Tridify
//
//  Created by Maged Alosali on 14/05/2023.
//

import SwiftUI

struct CustomCameraView: View {
    
    let cameraService = CameraService()
    
    @State var capturedImagesData = [Data]()
    @State var jpegRepresentationData = [Data]()
    @State private var showSheet = false
    @State private var showSendSheet = false
    @State private var showAlert = false
    @State private var deleteLast = false
    private let minImagesCount = 10
    private var sendFlag: Bool {
        capturedImagesData.count > minImagesCount
    }
    
    var body: some View {
        GeometryReader { geo in
            VStack {
                CameraView(cameraService: cameraService) { result in
                    switch result {
                    case .success(let photo):
                        if let data = photo.fileDataRepresentation() {
                            if let image = UIImage(data: data){
                                capturedImagesData.append(data)
                                if let jpegData = image.jpegData(compressionQuality: 1){
                                    jpegRepresentationData.append(jpegData)
                                    
                                }
                                else {
                                    print ("jpegRepresentation error")
                                }
                                
                            }
                            else {
                                print("Couldn't transform the data to image")
                            }
                        }
                        else {
                            print("Couldn't capture the image")
                        }
                    case .failure(let err):
                        print(err.localizedDescription)
                    }
                
                }
                VStack {
                    HStack {
                        Button {
                            if !sendFlag {
                                
                            }
                            else {
                                var imagesSize = 0.0
                                for data in jpegRepresentationData {
                                    imagesSize += Double(data.count)
                                }
                                imagesSize /= 1024
                                imagesSize /= 1024
                                print("Image Size \(imagesSize) MB")
                                
                            }
                            
                        } label: {
                            HStack {
                                Text("Send")
                                Image(systemName: "checkmark")
                            }
                            .font(.subheadline)
                            .kerning(2)
                            .foregroundColor(.black)
                            .frame(width: 100, height: 40)
                            .background(sendFlag ? .white : .gray)
                            .clipShape(
                                Capsule()
                            )
                        }
                        .disabled(!sendFlag)
                        Spacer()
                        Button {
                            cameraService.capturePhotos()
                        } label: {
                            Image (systemName: "circle")
                                .font(.system(size:72))
                                .foregroundColor(.white)
                        }
                        Spacer()
                        Group {
                            if (!capturedImagesData.isEmpty)  {
                                Button {
                                    showSheet.toggle()
                                } label: {
                                    Image(uiImage: UIImage(data: capturedImagesData[capturedImagesData.count - 1])!)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 80, height: 80)
                                        .clipShape(RoundedRectangle(cornerRadius: 20))
                                }
                            }
                            else {
                                RoundedRectangle(cornerRadius: 20)
                                    .frame(width: 80, height: 80)
                                    .foregroundColor(.emptyImage)
                            }
                        }
                        
                    }
                    .padding(.bottom)
                    .padding(.horizontal, 8)
                }
            }
        }
        .sheet(isPresented: $showSheet, onDismiss: {
            if (deleteLast){
                _ = capturedImagesData.popLast()
                deleteLast = false
        }
        }) {
            ImagesView(capturedData: $capturedImagesData, deleteLast: $deleteLast)
        }
        .alert("Take more captures", isPresented: $showAlert) {
            Button ("OK", role: .cancel) {}
        } message: {
            Text ("The limit of images to create 3D model is \(minImagesCount)")
        }
        .preferredColorScheme(.dark)
    }
}

struct CustomCameraView_Previews: PreviewProvider {
    static var previews: some View {
        CustomCameraView()
    }
}

