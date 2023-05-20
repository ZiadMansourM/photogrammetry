//
//  GalleryView.swift
//  Tridify
//
//  Created by Maged Alosali on 18/05/2023.
//

import SwiftUI
struct GalleryView: View {
    
    @State private var models = [ "Hulk", "Goal", "Nan"]
    @State private var isShowing = false
    @Binding private var cameraViewOn: Bool
    @AppStorage("onBoarding") var onBoarding = false
    var body: some View {
        NavigationView {
            VStack {
                HStack {
                    LogoView()
                        .frame(width: 50)
                    
                    Text ("Tridify")
                        .font(.largeTitle)
                        .kerning(12)
                    Spacer()
                    
                    Button {
                        cameraViewOn = true
                    } label: {
                        Image(systemName: "plus")
                            .fontWeight(.semibold)
                            .font(.title2)
                            .foregroundColor(.black)
                    }
                    
                    Button {
                        isShowing.toggle()
                    } label: {
                        Image(systemName: "power")
                            .fontWeight(.semibold)
                            .font(.title2)
                            .foregroundColor(.black)
                    }
                    .padding(.leading, 10)
                }
                .padding(10)
                
                ScrollView {
                    VStack() {
                        ForEach(models.indices, id: \.self){ i in
                            NavigationLink {
                                ModelSceneView(modelName: models[i])
                            } label: {
                                VStack {
                                    ModelSceneView(modelName: nil)
                                        .frame(height: 200)
                                    HStack {
                                        
                                        Text (models[i])
                                            .font(.title2)
                                        Spacer()
                                        Image(systemName: "chevron.forward")
                                    }
                                    .padding(.horizontal)
                                    .padding(.bottom)
                                    .foregroundColor(.black)
                                }
                                .clipShape(RoundedRectangle(cornerRadius: 20))
                                .overlay(content: {
                                    RoundedRectangle(cornerRadius: 20)
                                        .stroke(lineWidth: 2)
                                        .foregroundColor(.black)
                                })
                                .padding(10)
                            }
                            
                        }
                    }
                }
            }
        }
        .alert("Log out", isPresented: $isShowing) {
            HStack {
                Button("Log Out", role: .destructive){
                    onBoarding = true
                }
                Button("Cancel", role: .cancel){}
            }
        } message: {
            Text("Are you sure, you will lose all the data created?")
        }
    }
    
    init (cameraViewOn: Binding<Bool>){
        _cameraViewOn = cameraViewOn
    }
}

