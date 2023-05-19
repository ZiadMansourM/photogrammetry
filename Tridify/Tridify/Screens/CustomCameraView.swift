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
    @State var lastCapturedImage: UIImage? = nil
    @State private var showSheet = false
    @State private var deleteLast = false
    @Environment (\.colorScheme) private var colorScheme
    var body: some View {
        GeometryReader { geo in
            VStack {
                CameraView(cameraService: cameraService) { result in
                    switch result {
                    case .success(let photo):
                        if let data = photo.fileDataRepresentation() {
                            if let capturedImage = UIImage(data: data){
                                print("capturedImage size: \(capturedImagesData.count)")
                                capturedImagesData.append(data)
                                lastCapturedImage = capturedImage
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
                        Spacer()
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
                        .padding(.horizontal, 8)
                        
                    }
                    .padding(.bottom)
                }
            }
        }
        .sheet(isPresented: $showSheet, onDismiss: {
            if (deleteLast){
                _ = capturedImagesData.popLast()
            }
        }) {
            ImagesView(capturedData: $capturedImagesData, deleteLast: $deleteLast)
        }
        .preferredColorScheme(.dark)
    }
}

